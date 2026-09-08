"""Offline demo: a full run_cycle() against a fake broker and generated data.

    python -m tradebot.main --offline-demo

This exists because a cycle should be observable before it is trusted with
money, and because the real thing needs credentials and network access. The
broker is in-memory and the bars are generated, but everything else — the
reconciliation, the risk gates, the sizing, the order construction, the state
persistence — is the production code path.

It is a wiring check, not a strategy result. Nothing here says anything about
whether the strategy makes money.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradebot.alerts import Alerter
from tradebot.config import Config, Secrets, Settings
from tradebot.data import MarketData, Quote
from tradebot.execution import AccountSnapshot, BrokerOrder, BrokerPosition
from tradebot.logging_setup import get_logger

log = get_logger(__name__)


class OfflineBroker:
    """In-memory broker. Fills limit orders instantly at their limit price."""

    def __init__(self, equity: float = 50.0):
        self.equity = equity
        self.cash = equity
        self.positions: dict[str, BrokerPosition] = {}
        self.orders: dict[str, BrokerOrder] = {}
        self.submitted: list[BrokerOrder] = []
        self._next_id = 1

    def _new_id(self) -> str:
        self._next_id += 1
        return f"offline-{self._next_id}"

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=self.equity, cash=self.cash, buying_power=self.cash)

    def get_positions(self) -> list[BrokerPosition]:
        return list(self.positions.values())

    def get_open_orders(self) -> list[BrokerOrder]:
        return [o for o in self.orders.values() if o.is_open]

    def _record(self, order: BrokerOrder) -> BrokerOrder:
        self.orders[order.id] = order
        self.submitted.append(order)
        return order

    def submit_limit(self, symbol, side, qty, limit_price, client_order_id) -> BrokerOrder:
        order = BrokerOrder(
            id=self._new_id(),
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            status="filled",
            order_type="limit",
            limit_price=limit_price,
            filled_qty=qty,
            filled_avg_price=limit_price,
        )
        if side == "buy":
            self.cash -= qty * limit_price
            self.positions[symbol] = BrokerPosition(symbol, qty, limit_price, qty * limit_price)
        return self._record(order)

    def submit_stop_limit(
        self, symbol, side, qty, stop_price, limit_price, client_order_id
    ) -> BrokerOrder:
        # Rests at the exchange; does not fill in this demo.
        return self._record(
            BrokerOrder(
                id=self._new_id(),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                qty=qty,
                status="new",
                order_type="stop_limit",
                limit_price=limit_price,
            )
        )

    def submit_market(self, symbol, side, qty, client_order_id) -> BrokerOrder:
        position = self.positions.pop(symbol, None)
        price = position.avg_entry_price if position else 0.0
        if side == "sell":
            self.cash += qty * price
        return self._record(
            BrokerOrder(
                id=self._new_id(),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                qty=qty,
                status="filled",
                order_type="market",
                filled_qty=qty,
                filled_avg_price=price,
            )
        )

    def cancel_order(self, order_id: str) -> None:
        order = self.orders.get(order_id)
        if order:
            self.orders[order_id] = BrokerOrder(**{**order.__dict__, "status": "canceled"})


def demo_bars(symbols: list[str], now: datetime, days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generated bars ending today, with one symbol engineered to signal.

    Without a deliberate signal the demo would show an idle cycle, which
    exercises none of the sizing, risk or order-construction path.
    """
    rng = np.random.default_rng(seed)
    index = pd.DatetimeIndex(
        [(now - timedelta(days=days - 1 - i)).replace(hour=0, minute=0, second=0, microsecond=0)
         for i in range(days)],
        name="timestamp",
    )
    frames = []
    for offset, symbol in enumerate(symbols):
        base = 100.0 * (1 + offset * 0.5)
        # Gentle uptrend so the SMA filter can pass.
        drift = np.linspace(0, 0.35, days)
        noise = rng.normal(0, 0.01, days).cumsum()
        close = base * np.exp(drift + noise)
        high = close * 1.004
        low = close * 0.996
        volume = np.full(days, 1_000_000.0) * (1 + offset * 0.1)

        if offset == 0:
            # Engineer the final bar into the 2-4% pullback band on high volume.
            close[-1] = close[-2] * 0.99
            high[-2] = close[-1] / (1 - 0.03)
            high[-1] = close[-1] * 1.001
            low[-1] = close[-1] * 0.995
            volume[-1] = volume[0] * 3.0

        frame = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
        frame["symbol"] = symbol
        frames.append(
            frame.set_index("symbol", append=True).reorder_levels(["symbol", "timestamp"])
        )
    return pd.concat(frames).sort_index()


class OfflineMarketData(MarketData):
    """MarketData with the network calls replaced by generated series."""

    def __init__(self, config: Config, now: datetime):
        super().__init__(config, data_client=object(), trading_client=object())
        self._now = now
        self._symbols = config.settings.universe.resolve()[:6]
        self._bars = demo_bars(self._symbols, now)

    def discover_universe(self) -> list[str]:
        return self._symbols

    def get_daily_bars(self, symbols, start, end=None, force_refresh=False) -> pd.DataFrame:
        wanted = {s.upper() for s in symbols}
        mask = self._bars.index.get_level_values("symbol").isin(wanted)
        return self._bars[mask]

    def get_latest_quotes(self, symbols) -> dict[str, Quote]:
        quotes: dict[str, Quote] = {}
        for symbol in symbols:
            try:
                last = float(self._bars.xs(symbol, level="symbol")["close"].iloc[-1])
            except KeyError:
                continue
            quotes[symbol] = Quote(
                symbol=symbol,
                bid=last * 0.9995,
                ask=last * 1.0005,
                timestamp=self._now - timedelta(seconds=5),
            )
        return quotes


def run_offline_demo() -> int:
    from tradebot.main import TradingBot

    now = datetime.now(timezone.utc)
    settings = Settings.load()
    config = Config(secrets=Secrets(alpaca_paper=True), settings=settings)

    # Keep demo state out of the real state file.
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "demo_state.json"
        # Point state at a temp dir so the demo cannot clobber real bot state.
        setattr(type(settings), "state_path", property(lambda self: state_path))
        broker = OfflineBroker(equity=settings.risk.starting_capital)
        bot = TradingBot(
            config=config,
            market=OfflineMarketData(config, now),
            broker=broker,
            alerter=Alerter(config),  # no webhook configured -> alerts are no-ops
        )

        print("\n" + "=" * 74)
        print("OFFLINE DEMO — fake broker, generated bars. Not a strategy result.")
        print("=" * 74 + "\n")

        print(">>> CYCLE 1: flat, expect a signal, sizing, risk check and an entry\n")
        first = bot.run_cycle()

        print("\n>>> CYCLE 2: position now open, expect reconciliation and a resting stop\n")
        second = bot.run_cycle()

        print("\n" + "=" * 74)
        print("RESULT")
        print("=" * 74)
        for label, summary in (("cycle 1", first), ("cycle 2", second)):
            print(f"  {label}:")
            for action in summary["actions"] or ["(none)"]:
                print(f"      action     {action}")
            for rejection in summary["rejections"][:4]:
                print(f"      rejected   {rejection}")
            for error in summary["errors"]:
                print(f"      ERROR      {error}")
        print(f"\n  broker positions   {list(broker.positions)}")
        print(f"  broker cash        ${broker.cash:,.4f}")
        print("  orders submitted:")
        for order in broker.submitted:
            price = order.limit_price if order.limit_price is not None else 0.0
            print(
                f"      {order.order_type:<11} {order.side:<4} {order.symbol:<10} "
                f"qty={order.qty:<14.8f} px={price:<12.4f} {order.client_order_id}"
            )
        print("=" * 74)
    return 0
