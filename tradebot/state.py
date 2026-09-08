"""Persisted bot state: positions, orders, loss anchors, halts.

Alpaca is the source of truth for positions and orders; this file holds what
the broker cannot tell us — the equity we started the day and week with, how
many round trips we have done today, why we are halted, and which cycle each
order belongs to.

Written atomically (temp file then rename) so a process killed mid-write leaves
the previous state intact rather than a truncated file. A bot that loses its
day anchor forgets it was down 4% and gets a fresh 5% of rope.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tradebot.logging_setup import get_logger
from tradebot.strategy import Position

log = get_logger(__name__)

STATE_VERSION = 1


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def week_key(moment: datetime) -> str:
    """ISO year-week, the identity of the weekly loss anchor."""
    iso = moment.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def day_key(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).date().isoformat()


@dataclass
class OrderRecord:
    """An order the bot submitted, tracked until it fills or is cancelled."""

    client_order_id: str
    symbol: str
    side: str
    qty: float
    limit_price: float
    intent: str
    submitted_cycle: int
    submitted_at: str
    broker_order_id: str | None = None
    status: str = "new"

    def cycles_open(self, current_cycle: int) -> int:
        return max(0, current_cycle - self.submitted_cycle)


@dataclass
class PositionRecord:
    symbol: str
    qty: float
    entry_price: float
    entry_date: str
    stop_order_id: str | None = None

    def to_position(self) -> Position:
        import pandas as pd

        return Position(
            symbol=self.symbol,
            qty=self.qty,
            entry_price=self.entry_price,
            entry_date=pd.Timestamp(self.entry_date),
        )


@dataclass
class BotState:
    """Everything that must survive a restart."""

    version: int = STATE_VERSION
    positions: dict[str, PositionRecord] = field(default_factory=dict)
    open_orders: dict[str, OrderRecord] = field(default_factory=dict)

    day_anchor_equity: float = 0.0
    day_anchor_key: str = ""
    week_anchor_equity: float = 0.0
    week_anchor_key: str = ""

    round_trips_today: int = 0
    round_trip_day_key: str = ""

    halted_until: str | None = None
    halt_reason: str = ""

    cycle: int = 0
    last_cycle_at: str | None = None
    realized_pnl_today: float = 0.0

    # ------------------------------------------------------------ persistence

    @classmethod
    def load(cls, path: Path | str) -> "BotState":
        """Read state, or return a fresh one. A corrupt file is never fatal —
        reconciliation against Alpaca rebuilds positions anyway."""
        path = Path(path)
        if not path.exists():
            log.info("state.new", path=str(path))
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("state.corrupt", path=str(path), error=str(exc), action="starting fresh")
            return cls()

        state = cls(
            version=raw.get("version", STATE_VERSION),
            day_anchor_equity=raw.get("day_anchor_equity", 0.0),
            day_anchor_key=raw.get("day_anchor_key", ""),
            week_anchor_equity=raw.get("week_anchor_equity", 0.0),
            week_anchor_key=raw.get("week_anchor_key", ""),
            round_trips_today=raw.get("round_trips_today", 0),
            round_trip_day_key=raw.get("round_trip_day_key", ""),
            halted_until=raw.get("halted_until"),
            halt_reason=raw.get("halt_reason", ""),
            cycle=raw.get("cycle", 0),
            last_cycle_at=raw.get("last_cycle_at"),
            realized_pnl_today=raw.get("realized_pnl_today", 0.0),
        )
        state.positions = {
            symbol: PositionRecord(**record)
            for symbol, record in raw.get("positions", {}).items()
        }
        state.open_orders = {
            order_id: OrderRecord(**record)
            for order_id, record in raw.get("open_orders", {}).items()
        }
        return state

    def save(self, path: Path | str) -> None:
        """Atomic write: a process killed mid-save leaves the old file intact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        )
        try:
            with handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, path)
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise

    # ---------------------------------------------------------------- anchors

    def roll_anchors(self, now: datetime, equity: float) -> dict[str, Any]:
        """Reset the daily/weekly loss anchors when their period turns over.

        24/7 markets have no session open, so the anchors turn at UTC midnight
        and at Monday UTC midnight.
        """
        rolled: dict[str, Any] = {}
        today, this_week = day_key(now), week_key(now)

        if self.day_anchor_key != today:
            self.day_anchor_key = today
            self.day_anchor_equity = equity
            self.round_trips_today = 0
            self.round_trip_day_key = today
            self.realized_pnl_today = 0.0
            rolled["day_anchor"] = equity

        if self.week_anchor_key != this_week:
            self.week_anchor_key = this_week
            self.week_anchor_equity = equity
            rolled["week_anchor"] = equity

        # Defensive: a first run has no anchors at all.
        if self.day_anchor_equity <= 0:
            self.day_anchor_equity = equity
        if self.week_anchor_equity <= 0:
            self.week_anchor_equity = equity
        return rolled

    # ----------------------------------------------------------------- halts

    @property
    def halted_until_dt(self) -> datetime | None:
        return _parse(self.halted_until)

    def set_halt(self, until: datetime | None, reason: str) -> None:
        self.halted_until = _iso(until)
        self.halt_reason = reason

    def clear_halt(self) -> None:
        self.halted_until = None
        self.halt_reason = ""

    def is_halted(self, now: datetime) -> bool:
        until = self.halted_until_dt
        return until is not None and now < until

    # ------------------------------------------------------------- positions

    def as_positions(self) -> list[Position]:
        return [record.to_position() for record in self.positions.values()]

    def record_entry(
        self, symbol: str, qty: float, entry_price: float, entry_date: datetime
    ) -> None:
        self.positions[symbol] = PositionRecord(
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            entry_date=entry_date.isoformat(),
        )

    def record_exit(self, symbol: str, exit_price: float, now: datetime) -> float:
        """Remove a position and return its realized P&L."""
        record = self.positions.pop(symbol, None)
        if record is None:
            return 0.0
        pnl = (exit_price - record.entry_price) * record.qty
        self.realized_pnl_today += pnl
        today = day_key(now)
        if self.round_trip_day_key != today:
            self.round_trip_day_key = today
            self.round_trips_today = 0
        self.round_trips_today += 1
        return pnl

    # ---------------------------------------------------------------- orders

    def add_order(self, record: OrderRecord) -> None:
        self.open_orders[record.client_order_id] = record

    def drop_order(self, client_order_id: str) -> None:
        self.open_orders.pop(client_order_id, None)

    def stale_orders(self, current_cycle: int, max_cycles: int) -> list[OrderRecord]:
        """Entry orders unfilled for longer than the configured cycle budget.

        Only entries: an unfilled protective stop must never be cancelled for
        being old.
        """
        return [
            record
            for record in self.open_orders.values()
            if record.intent == "entry" and record.cycles_open(current_cycle) >= max_cycles
        ]

    def touch(self, now: datetime) -> None:
        self.cycle += 1
        self.last_cycle_at = _iso(now)
