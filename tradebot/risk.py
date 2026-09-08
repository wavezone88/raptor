"""Risk manager: the last gate before any order reaches the broker.

Every proposed order goes through RiskManager.check(), which returns an
approved order, a resized order, or a rejection carrying the reason. Nothing in
execution.py may bypass it.

Two principles hold throughout:

  * Sells are never blocked. Every rule here can stop the bot opening or adding
    risk; none of them can stop it reducing risk. A rule that could veto an
    exit would be able to trap the bot in a losing position.
  * Rejections say why. The reason string is alerted and logged verbatim, so a
    bot that quietly stops trading can be diagnosed from the log alone.

Crypto notes: the pattern-day-trader guard from the equity spec does not apply
(FINRA round-trip rules cover margin securities accounts, not crypto), and
settlement is instant so there is no unsettled cash. max_round_trips_per_day
replaces PDT as a churn and fee brake, and the daily/weekly loss anchors are
tied to UTC boundaries because a 24/7 market has no session open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Literal

from tradebot.config import RiskSettings, get_config
from tradebot.strategy import Position

OrderSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class SymbolStats:
    """What the risk manager needs to know about a symbol to vet a trade."""

    symbol: str
    last_price: float
    avg_dollar_volume_30d: float


@dataclass(frozen=True)
class ProposedOrder:
    """An order the bot wants to place, before risk has seen it."""

    symbol: str
    side: OrderSide
    qty: float
    limit_price: float
    intent: str = ""          # "entry", "stop", "target", "time_stop", "flatten"
    # The stop this entry will be protected by. Sizing divides the risk budget
    # by the ACTUAL distance to it, so a wider ATR stop buys proportionally
    # less. Without this an ATR stop would silently multiply risk per trade.
    stop_price: float | None = None

    @property
    def notional(self) -> float:
        return self.qty * self.limit_price

    @property
    def stop_distance_pct(self) -> float | None:
        if self.stop_price is None or self.limit_price <= 0:
            return None
        distance = (self.limit_price - self.stop_price) / self.limit_price
        return distance if distance > 0 else None

    def resized_to(self, qty: float) -> "ProposedOrder":
        return ProposedOrder(
            self.symbol, self.side, qty, self.limit_price, self.intent, self.stop_price
        )


@dataclass(frozen=True)
class AccountState:
    """Everything risk needs about the account, gathered once per cycle.

    A plain snapshot rather than a live client, so every rule is a pure
    function of its inputs and testable without a broker.
    """

    equity: float
    cash: float
    buying_power: float
    positions: list[Position] = field(default_factory=list)
    symbol_stats: dict[str, SymbolStats] = field(default_factory=dict)
    day_anchor_equity: float = 0.0
    week_anchor_equity: float = 0.0
    round_trips_today: int = 0
    halted_until: datetime | None = None
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def position_for(self, symbol: str) -> Position | None:
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None

    @property
    def open_position_count(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class RiskDecision:
    """The verdict. `order` is None when rejected."""

    approved: bool
    reason: str
    order: ProposedOrder | None = None
    resized: bool = False

    @classmethod
    def reject(cls, reason: str) -> "RiskDecision":
        return cls(approved=False, reason=reason)

    @classmethod
    def approve(cls, order: ProposedOrder, reason: str = "ok", resized: bool = False) -> "RiskDecision":
        return cls(approved=True, reason=reason, order=order, resized=resized)


@dataclass(frozen=True)
class HaltDecision:
    """Whether trading is halted, and whether we must flatten to get there."""

    halted: bool
    reason: str = ""
    flatten: bool = False
    halted_until: datetime | None = None


def floor_to(value: float, decimals: int) -> float:
    """Round DOWN to `decimals` places.

    Always down: rounding a position size up would breach the very limit the
    size was derived from.
    """
    if value <= 0:
        return 0.0
    factor = 10**decimals
    return math.floor(value * factor) / factor


def next_utc_midnight(now: datetime) -> datetime:
    """Start of the next UTC day — the 24/7 stand-in for 'the next session'."""
    return datetime.combine((now + timedelta(days=1)).date(), time.min, tzinfo=timezone.utc)


def next_utc_monday(now: datetime) -> datetime:
    """Start of the next Monday, UTC. Always strictly in the future."""
    days_ahead = (7 - now.weekday()) % 7 or 7
    return datetime.combine(
        (now + timedelta(days=days_ahead)).date(), time.min, tzinfo=timezone.utc
    )


def position_size(
    equity: float,
    price: float,
    settings: RiskSettings,
    available_cash: float | None = None,
    stop_distance_pct: float | None = None,
) -> float:
    """Units to buy: (equity x risk%) / stop distance, then capped.

    Caps, in order:
      1. risk budget    — equity * risk_per_trade_pct / stop distance
      2. concentration  — equity * max_position_pct_equity
      3. cash on hand   — crypto is non-marginable, so cash is the hard ceiling

    `stop_distance_pct` is the ACTUAL distance to this order's stop. Under
    atr_multiple mode it varies per symbol, and using the configured default
    instead would size every trade as though its stop were 4% away — turning a
    wider stop into proportionally more risk, which is the opposite of the
    point. Falls back to the configured default when not supplied.

    Rounded down to fractional precision.
    """
    if price <= 0 or equity <= 0:
        return 0.0
    distance = stop_distance_pct if stop_distance_pct and stop_distance_pct > 0 else settings.stop_distance_pct
    risk_budget = equity * settings.risk_per_trade_pct
    notional = risk_budget / distance
    notional = min(notional, equity * settings.max_position_pct_equity)
    if available_cash is not None:
        notional = min(notional, max(0.0, available_cash))
    return floor_to(notional / price, settings.fractional_qty_decimals)


class RiskManager:
    """Stateless given an AccountState. All rules are configurable."""

    def __init__(self, settings: RiskSettings | None = None, min_order_notional: float = 1.0):
        self.settings = settings or get_config().settings.risk
        self.min_order_notional = min_order_notional

    # ------------------------------------------------------------------ halts

    def drawdown_from(self, anchor: float, equity: float) -> float:
        """Fractional loss against an anchor. Positive means down."""
        if anchor <= 0:
            return 0.0
        return (anchor - equity) / anchor

    def evaluate_halts(self, state: AccountState) -> HaltDecision:
        """Daily and weekly loss limits, and any halt already in force.

        Checked before anything else in the cycle. When this returns
        flatten=True the bot closes everything and stops trading until
        halted_until, rather than merely declining new entries.
        """
        if state.halted_until is not None and state.now < state.halted_until:
            return HaltDecision(
                halted=True,
                reason=f"halted until {state.halted_until.isoformat()}",
                flatten=False,
                halted_until=state.halted_until,
            )

        weekly_loss = self.drawdown_from(state.week_anchor_equity, state.equity)
        if weekly_loss >= self.settings.weekly_loss_limit_pct:
            until = next_utc_monday(state.now)
            return HaltDecision(
                halted=True,
                reason=(
                    f"weekly loss limit: down {weekly_loss:.2%} from Monday anchor "
                    f"{state.week_anchor_equity:.2f}, limit "
                    f"{self.settings.weekly_loss_limit_pct:.2%}"
                ),
                flatten=True,
                halted_until=until,
            )

        daily_loss = self.drawdown_from(state.day_anchor_equity, state.equity)
        if daily_loss >= self.settings.daily_loss_limit_pct:
            until = next_utc_midnight(state.now)
            return HaltDecision(
                halted=True,
                reason=(
                    f"daily loss limit: down {daily_loss:.2%} from day anchor "
                    f"{state.day_anchor_equity:.2f}, limit "
                    f"{self.settings.daily_loss_limit_pct:.2%}"
                ),
                flatten=True,
                halted_until=until,
            )

        return HaltDecision(halted=False)

    # ------------------------------------------------------------------ check

    def check(self, order: ProposedOrder, state: AccountState) -> RiskDecision:
        """Approve, resize, or reject a proposed order."""
        if order.qty <= 0:
            return RiskDecision.reject(f"{order.symbol}: non-positive qty {order.qty}")
        if order.limit_price <= 0:
            return RiskDecision.reject(f"{order.symbol}: non-positive price {order.limit_price}")

        if order.side == "sell":
            return self._check_sell(order, state)
        return self._check_buy(order, state)

    def _check_sell(self, order: ProposedOrder, state: AccountState) -> RiskDecision:
        """Exits are never blocked — only clamped to what we actually hold."""
        position = state.position_for(order.symbol)
        if position is None:
            return RiskDecision.reject(f"{order.symbol}: no open position to sell")
        if order.qty > position.qty:
            clamped = floor_to(position.qty, self.settings.fractional_qty_decimals)
            if clamped <= 0:
                return RiskDecision.reject(f"{order.symbol}: position too small to sell")
            return RiskDecision.approve(
                order.resized_to(clamped),
                reason=f"clamped sell to held qty {clamped}",
                resized=True,
            )
        return RiskDecision.approve(order)

    def _check_buy(self, order: ProposedOrder, state: AccountState) -> RiskDecision:
        halt = self.evaluate_halts(state)
        if halt.halted:
            return RiskDecision.reject(f"{order.symbol}: {halt.reason}")

        if state.round_trips_today >= self.settings.max_round_trips_per_day:
            return RiskDecision.reject(
                f"{order.symbol}: round-trip limit reached "
                f"({state.round_trips_today}/{self.settings.max_round_trips_per_day} today)"
            )

        held = state.position_for(order.symbol)
        if held is None and state.open_position_count >= self.settings.max_open_positions:
            return RiskDecision.reject(
                f"{order.symbol}: max open positions "
                f"({state.open_position_count}/{self.settings.max_open_positions})"
            )

        if held is not None and order.limit_price < held.entry_price:
            return RiskDecision.reject(
                f"{order.symbol}: never add to a losing position "
                f"(price {order.limit_price:.6g} < entry {held.entry_price:.6g})"
            )

        stats = state.symbol_stats.get(order.symbol)
        if stats is None:
            return RiskDecision.reject(f"{order.symbol}: no liquidity stats available")
        if stats.avg_dollar_volume_30d < self.settings.min_avg_dollar_volume_30d:
            return RiskDecision.reject(
                f"{order.symbol}: 30d dollar volume {stats.avg_dollar_volume_30d:,.0f} "
                f"below floor {self.settings.min_avg_dollar_volume_30d:,.0f}"
            )
        if stats.last_price < self.settings.min_price:
            return RiskDecision.reject(
                f"{order.symbol}: price {stats.last_price:.6g} below floor "
                f"{self.settings.min_price:.6g}"
            )

        # Crypto is non-marginable: cash, not buying_power, is the ceiling.
        available = min(state.cash, state.buying_power)
        if available <= 0:
            return RiskDecision.reject(
                f"{order.symbol}: no available cash (cash={state.cash:.2f}, "
                f"buying_power={state.buying_power:.2f})"
            )

        allowed_qty = position_size(
            state.equity,
            order.limit_price,
            self.settings,
            available,
            stop_distance_pct=order.stop_distance_pct,
        )
        if allowed_qty <= 0:
            return RiskDecision.reject(
                f"{order.symbol}: sized to zero at price {order.limit_price:.6g} "
                f"(equity {state.equity:.2f}, cash {available:.2f})"
            )

        final_qty = min(order.qty, allowed_qty)
        final_qty = floor_to(final_qty, self.settings.fractional_qty_decimals)
        notional = final_qty * order.limit_price

        if notional < self.min_order_notional:
            return RiskDecision.reject(
                f"{order.symbol}: notional {notional:.2f} below broker minimum "
                f"{self.min_order_notional:.2f}"
            )

        if final_qty < order.qty:
            return RiskDecision.approve(
                order.resized_to(final_qty),
                reason=(
                    f"resized {order.qty:.8f} -> {final_qty:.8f} "
                    f"(notional {notional:.2f})"
                ),
                resized=True,
            )
        return RiskDecision.approve(order, reason=f"ok (notional {notional:.2f})")
