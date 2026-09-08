"""Tests for the backtest simulation loop.

The simulation is the only place the strategy and the risk manager are wired
together, so these pin the wiring: no look-ahead, risk rules actually applied,
and costs actually deducted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradebot.backtest import Trade, max_drawdown, simulate, synthetic_bars, trade_statistics
from tradebot.config import Config, Secrets, Settings

from .conftest import START


@pytest.fixture
def config():
    return Config(secrets=Secrets(), settings=Settings.load())


def test_synthetic_bars_are_deterministic():
    """The pipeline check must be reproducible or it proves nothing."""
    first = synthetic_bars(["BTC/USD", "ETH/USD"], days=100, seed=7)
    second = synthetic_bars(["BTC/USD", "ETH/USD"], days=100, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_paths():
    a = synthetic_bars(["BTC/USD"], days=100, seed=1)
    b = synthetic_bars(["BTC/USD"], days=100, seed=2)
    assert not a["close"].equals(b["close"])


def test_simulation_never_exceeds_max_open_positions(config):
    """The risk manager's position cap must bind inside the simulation, not
    only in unit tests of risk.py."""
    bars = synthetic_bars(config.settings.universe.resolve(), days=400, seed=11)
    result = simulate(bars, config)

    open_qty: dict[str, float] = {}
    peak = 0
    for order in sorted(result.orders, key=lambda o: o["timestamp"]):
        symbol = order["symbol"]
        open_qty[symbol] = open_qty.get(symbol, 0.0) + order["size"]
        if abs(open_qty[symbol]) < 1e-12:
            open_qty.pop(symbol)
        peak = max(peak, len(open_qty))
    assert peak <= config.settings.risk.max_open_positions


def test_simulation_never_spends_more_than_it_has(config):
    """Cash must never go negative — crypto is non-marginable."""
    bars = synthetic_bars(config.settings.universe.resolve(), days=400, seed=12)
    result = simulate(bars, config)
    assert (result.equity_curve > 0).all(), "equity went non-positive"


def test_costs_reduce_return(config):
    """The whole point of the cost assumption: it must actually be charged."""
    bars = synthetic_bars(config.settings.universe.resolve(), days=400, seed=13)
    net = simulate(bars, config)
    gross = simulate(bars, config, cost=0.0)
    assert gross.equity_curve.iloc[-1] > net.equity_curve.iloc[-1]


def test_entries_fill_at_the_next_bar_open_not_the_signal_bar_close(config):
    """No look-ahead: an entry cannot be priced at the close that produced it."""
    bars = synthetic_bars(config.settings.universe.resolve(), days=300, seed=14)
    result = simulate(bars, config)
    entries = [o for o in result.orders if o["intent"] == "entry"]
    assert entries, "no entries generated; test proves nothing"
    for order in entries[:50]:
        bar = bars.loc[(order["symbol"], order["timestamp"])]
        assert order["price"] == pytest.approx(float(bar["open"]))


def test_trade_pnl_accounts_for_both_sides_of_cost():
    trade = Trade(
        symbol="BTC/USD",
        qty=1.0,
        entry_date=START,
        entry_price=100.0,
        exit_date=START,
        exit_price=110.0,
        reason="target",
    )
    assert trade.pnl(0.0) == pytest.approx(10.0)
    # 0.25% charged on both the $100 entry and the $110 exit.
    assert trade.pnl(0.0025) == pytest.approx(10.0 - (100.0 + 110.0) * 0.0025)


def test_trade_statistics_split_wins_and_losses():
    def trade(exit_price):
        return Trade("BTC/USD", 1.0, START, 100.0, START, exit_price, "target")

    stats = trade_statistics([trade(108.0), trade(96.0), trade(108.0)], 0.0)
    assert stats["trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["avg_win"] == pytest.approx(0.08)
    assert stats["avg_loss"] == pytest.approx(-0.04)
    assert stats["payoff_ratio"] == pytest.approx(2.0)


def test_trade_statistics_handle_no_trades():
    stats = trade_statistics([], 0.0)
    assert stats["trades"] == 0


def test_max_drawdown_is_negative_and_peak_relative():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(equity) == pytest.approx(-0.5)
