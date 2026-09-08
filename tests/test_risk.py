"""Tests for risk.py.

One test per rule, each constructed so that DELETING the rule from risk.py
makes it fail — not merely so that the current code passes. Where a rule has a
threshold, both sides of it are tested.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradebot.risk import (
    AccountState,
    HaltDecision,
    ProposedOrder,
    RiskManager,
    SymbolStats,
    floor_to,
    next_utc_midnight,
    next_utc_monday,
    position_size,
)
from tradebot.strategy import Position

NOW = datetime(2025, 6, 11, 15, 30, tzinfo=timezone.utc)  # a Wednesday
LIQUID = 50_000_000.0


def stats(symbol="BTC/USD", price=100.0, dollar_volume=LIQUID):
    return SymbolStats(symbol=symbol, last_price=price, avg_dollar_volume_30d=dollar_volume)


def account(
    equity=50.0,
    cash=50.0,
    buying_power=None,
    positions=None,
    symbol_stats=None,
    day_anchor=50.0,
    week_anchor=50.0,
    round_trips=0,
    halted_until=None,
    now=NOW,
):
    return AccountState(
        equity=equity,
        cash=cash,
        buying_power=cash if buying_power is None else buying_power,
        positions=positions or [],
        symbol_stats=symbol_stats or {"BTC/USD": stats(), "ETH/USD": stats("ETH/USD")},
        day_anchor_equity=day_anchor,
        week_anchor_equity=week_anchor,
        round_trips_today=round_trips,
        halted_until=halted_until,
        now=now,
    )


def buy(symbol="BTC/USD", qty=0.1, price=100.0):
    return ProposedOrder(symbol, "buy", qty, price, intent="entry")


def sell(symbol="BTC/USD", qty=0.1, price=100.0):
    return ProposedOrder(symbol, "sell", qty, price, intent="stop")


def held(symbol="BTC/USD", qty=0.1, entry=100.0):
    return Position(symbol, qty, entry, NOW - timedelta(days=1))


# ------------------------------------------------------------ position sizing


def test_position_size_is_risk_budget_over_stop_distance(risk_settings):
    """(equity x 1%) / 4% stop = 25% of equity in notional."""
    qty = position_size(equity=1_000.0, price=100.0, settings=risk_settings)
    assert qty * 100.0 == pytest.approx(250.0)


def test_position_size_respects_the_forty_percent_concentration_cap(risk_settings):
    """When the risk budget implies more than 40% of equity, the cap binds."""
    aggressive = risk_settings.model_copy(update={"risk_per_trade_pct": 0.05})
    # 5% risk / 4% stop = 125% of equity, far above the 40% cap.
    qty = position_size(equity=1_000.0, price=100.0, settings=aggressive)
    assert qty * 100.0 == pytest.approx(400.0)


def test_position_size_never_exceeds_available_cash(risk_settings):
    """Crypto is non-marginable; cash is a hard ceiling."""
    qty = position_size(equity=1_000.0, price=100.0, settings=risk_settings, available_cash=60.0)
    assert qty * 100.0 == pytest.approx(60.0, abs=0.01)


def test_position_size_rounds_down_not_up(risk_settings):
    """Rounding up would breach the limit the size was derived from."""
    coarse = risk_settings.model_copy(update={"fractional_qty_decimals": 2})
    qty = position_size(equity=1_000.0, price=99.0, settings=coarse)
    assert qty == 2.52  # 250/99 = 2.5252... floored, not rounded to 2.53
    assert qty * 99.0 <= 250.0


def test_position_size_is_zero_for_nonsense_inputs(risk_settings):
    assert position_size(0.0, 100.0, risk_settings) == 0.0
    assert position_size(1_000.0, 0.0, risk_settings) == 0.0


def test_floor_to_rounds_down():
    assert floor_to(1.239, 2) == 1.23
    assert floor_to(1.999, 0) == 1.0
    assert floor_to(-5.0, 2) == 0.0


# --------------------------------------------------------- concentration cap


def test_order_is_resized_down_to_the_risk_limit(risk_settings):
    """An oversized proposal is resized, not rejected — we still want the trade."""
    manager = RiskManager(risk_settings)
    decision = manager.check(buy(qty=100.0, price=100.0), account(equity=50.0, cash=50.0))
    assert decision.approved
    assert decision.resized
    assert decision.order.notional == pytest.approx(12.5, abs=0.01)


def test_single_position_cannot_exceed_forty_percent_of_equity(risk_settings):
    aggressive = risk_settings.model_copy(update={"risk_per_trade_pct": 0.05})
    manager = RiskManager(aggressive)
    decision = manager.check(
        buy(qty=1_000.0, price=1.0), account(equity=1_000.0, cash=1_000.0)
    )
    assert decision.approved
    assert decision.order.notional <= 1_000.0 * 0.40 + 1e-6


# ------------------------------------------------------------- position count


def test_fourth_position_is_rejected(risk_settings):
    manager = RiskManager(risk_settings)
    positions = [held("BTC/USD"), held("ETH/USD"), held("SOL/USD")]
    decision = manager.check(buy("AVAX/USD"), account(positions=positions))
    assert not decision.approved
    assert "max open positions" in decision.reason


def test_third_position_is_allowed(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(
        positions=[held("BTC/USD"), held("ETH/USD")],
        symbol_stats={"SOL/USD": stats("SOL/USD")},
    )
    assert manager.check(buy("SOL/USD"), state).approved


def test_adding_to_an_existing_position_ignores_the_position_cap(risk_settings):
    """At the cap we may still adjust a position we already hold."""
    manager = RiskManager(risk_settings)
    positions = [held("BTC/USD", entry=50.0), held("ETH/USD"), held("SOL/USD")]
    decision = manager.check(buy("BTC/USD", price=100.0), account(positions=positions))
    assert decision.approved, decision.reason


# ------------------------------------------------------- never add to a loser


def test_never_add_to_a_losing_position(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(positions=[held("BTC/USD", entry=100.0)])
    decision = manager.check(buy("BTC/USD", price=90.0), state)
    assert not decision.approved
    assert "never add to a losing position" in decision.reason


def test_adding_to_a_winning_position_is_allowed(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(positions=[held("BTC/USD", entry=100.0)])
    assert manager.check(buy("BTC/USD", price=110.0), state).approved


# ------------------------------------------------------------- daily loss cap


def test_daily_loss_limit_halts_and_flattens(risk_settings):
    """Down 5% from the day's anchor -> flatten everything, halt to next day."""
    manager = RiskManager(risk_settings)
    halt = manager.evaluate_halts(account(equity=47.5, day_anchor=50.0))
    assert halt.halted and halt.flatten
    assert "daily loss limit" in halt.reason
    assert halt.halted_until == next_utc_midnight(NOW)


def test_daily_loss_just_inside_the_limit_does_not_halt(risk_settings):
    manager = RiskManager(risk_settings)
    assert not manager.evaluate_halts(account(equity=47.6, day_anchor=50.0)).halted


def test_daily_loss_limit_blocks_new_buys(risk_settings):
    manager = RiskManager(risk_settings)
    decision = manager.check(buy(), account(equity=47.5, cash=47.5, day_anchor=50.0))
    assert not decision.approved
    assert "daily loss limit" in decision.reason


# ------------------------------------------------------------ weekly loss cap


def test_weekly_loss_limit_halts_until_next_monday(risk_settings):
    manager = RiskManager(risk_settings)
    halt = manager.evaluate_halts(account(equity=45.0, day_anchor=45.0, week_anchor=50.0))
    assert halt.halted and halt.flatten
    assert "weekly loss limit" in halt.reason
    assert halt.halted_until == next_utc_monday(NOW)
    assert halt.halted_until.weekday() == 0


def test_weekly_loss_just_inside_the_limit_does_not_halt(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(equity=45.1, day_anchor=45.1, week_anchor=50.0)
    assert not manager.evaluate_halts(state).halted


def test_weekly_limit_takes_precedence_over_daily(risk_settings):
    """Both breached: the longer halt must win, or the weekly halt is lost."""
    manager = RiskManager(risk_settings)
    halt = manager.evaluate_halts(account(equity=40.0, day_anchor=50.0, week_anchor=50.0))
    assert "weekly loss limit" in halt.reason
    assert halt.halted_until == next_utc_monday(NOW)


# ------------------------------------------------------------- existing halts


def test_an_active_halt_blocks_buys(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(halted_until=NOW + timedelta(hours=5))
    decision = manager.check(buy(), state)
    assert not decision.approved
    assert "halted until" in decision.reason


def test_an_expired_halt_does_not_block(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(halted_until=NOW - timedelta(hours=1))
    assert manager.check(buy(), state).approved


def test_an_active_halt_never_blocks_a_sell(risk_settings):
    """Halted means stop opening risk, not stop closing it."""
    manager = RiskManager(risk_settings)
    state = account(positions=[held()], halted_until=NOW + timedelta(hours=5))
    assert manager.check(sell(), state).approved


# ------------------------------------------------------- round-trip / churn


def test_fourth_round_trip_of_the_day_is_refused(risk_settings):
    """Replaces the PDT guard: crypto has no round-trip rule, but at ~0.25%
    per side unchecked churn is a guaranteed loss."""
    manager = RiskManager(risk_settings)
    decision = manager.check(buy(), account(round_trips=3))
    assert not decision.approved
    assert "round-trip limit" in decision.reason


def test_third_round_trip_of_the_day_is_allowed(risk_settings):
    manager = RiskManager(risk_settings)
    assert manager.check(buy(), account(round_trips=2)).approved


def test_round_trip_limit_never_blocks_a_sell(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(positions=[held()], round_trips=99)
    assert manager.check(sell(), state).approved


# ----------------------------------------------------------------- liquidity


def test_thin_symbol_is_rejected(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(symbol_stats={"SHIB/USD": stats("SHIB/USD", dollar_volume=1_000.0)})
    decision = manager.check(buy("SHIB/USD"), state)
    assert not decision.approved
    assert "dollar volume" in decision.reason


def test_symbol_just_above_the_liquidity_floor_is_accepted(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(
        symbol_stats={"SHIB/USD": stats("SHIB/USD", dollar_volume=10_000_001.0)}
    )
    assert manager.check(buy("SHIB/USD"), state).approved


def test_price_floor_is_enforced_when_configured(risk_settings):
    """Disabled by default for crypto, but the rule must still work."""
    with_floor = risk_settings.model_copy(update={"min_price": 10.0})
    manager = RiskManager(with_floor)
    state = account(symbol_stats={"SHIB/USD": stats("SHIB/USD", price=0.5)})
    decision = manager.check(buy("SHIB/USD", price=0.5), state)
    assert not decision.approved
    assert "below floor" in decision.reason


def test_missing_liquidity_stats_is_a_rejection_not_an_assumption(risk_settings):
    """Unknown liquidity must never be treated as acceptable liquidity."""
    manager = RiskManager(risk_settings)
    decision = manager.check(buy("DOGE/USD"), account(symbol_stats={}))
    assert not decision.approved
    assert "no liquidity stats" in decision.reason


# ---------------------------------------------------------------------- cash


def test_no_cash_means_no_buy(risk_settings):
    manager = RiskManager(risk_settings)
    decision = manager.check(buy(), account(equity=50.0, cash=0.0))
    assert not decision.approved
    assert "no available cash" in decision.reason


def test_buy_is_capped_by_cash_not_equity(risk_settings):
    """Equity includes open positions; only cash can fund a new one."""
    manager = RiskManager(risk_settings)
    decision = manager.check(buy(qty=10.0, price=100.0), account(equity=1_000.0, cash=5.0))
    assert decision.approved
    assert decision.order.notional <= 5.0 + 1e-9


def test_buying_power_lower_than_cash_is_the_binding_constraint(risk_settings):
    manager = RiskManager(risk_settings)
    decision = manager.check(
        buy(qty=10.0, price=100.0), account(equity=1_000.0, cash=500.0, buying_power=3.0)
    )
    assert decision.approved
    assert decision.order.notional <= 3.0 + 1e-9


# ------------------------------------------------------------ broker minimum


def test_order_below_broker_minimum_notional_is_rejected(risk_settings):
    """At $50 equity a sized position can fall under Alpaca's $1 floor."""
    manager = RiskManager(risk_settings, min_order_notional=1.0)
    # Anchors match equity so the loss limits stay quiet and this rule is the
    # only one that can fire. $2 equity sizes to $0.50 notional, under the floor.
    state = account(equity=2.0, cash=2.0, day_anchor=2.0, week_anchor=2.0)
    decision = manager.check(buy(qty=1.0, price=100.0), state)
    assert not decision.approved
    assert "below broker minimum" in decision.reason


# ---------------------------------------------------------------------- sells


def test_sell_without_a_position_is_rejected(risk_settings):
    manager = RiskManager(risk_settings)
    decision = manager.check(sell("BTC/USD"), account(positions=[]))
    assert not decision.approved
    assert "no open position" in decision.reason


def test_sell_larger_than_the_position_is_clamped(risk_settings):
    manager = RiskManager(risk_settings)
    state = account(positions=[held("BTC/USD", qty=0.05)])
    decision = manager.check(sell("BTC/USD", qty=5.0), state)
    assert decision.approved and decision.resized
    assert decision.order.qty == pytest.approx(0.05)


def test_sell_passes_every_gate_that_blocks_a_buy(risk_settings):
    """The single most important property here: risk can never trap us in a
    position. Every buy-side veto is active in this state simultaneously."""
    manager = RiskManager(risk_settings)
    state = account(
        equity=10.0,
        cash=0.0,
        positions=[held("BTC/USD"), held("ETH/USD"), held("SOL/USD")],
        symbol_stats={},
        day_anchor=50.0,
        week_anchor=50.0,
        round_trips=99,
        halted_until=NOW + timedelta(days=3),
    )
    assert manager.check(buy("BTC/USD"), state).approved is False
    assert manager.check(sell("BTC/USD"), state).approved is True


# --------------------------------------------------------------- input guards


@pytest.mark.parametrize("qty", [0.0, -1.0])
def test_non_positive_qty_is_rejected(risk_settings, qty):
    manager = RiskManager(risk_settings)
    assert not manager.check(buy(qty=qty), account()).approved


@pytest.mark.parametrize("price", [0.0, -5.0])
def test_non_positive_price_is_rejected(risk_settings, price):
    manager = RiskManager(risk_settings)
    assert not manager.check(buy(price=price), account()).approved


# ------------------------------------------------------------ session clocks


def test_next_utc_midnight_is_always_forward():
    assert next_utc_midnight(NOW) == datetime(2025, 6, 12, tzinfo=timezone.utc)


def test_next_utc_monday_from_midweek():
    assert next_utc_monday(NOW) == datetime(2025, 6, 16, tzinfo=timezone.utc)


def test_next_utc_monday_from_a_monday_is_a_week_out():
    """Never returns today, or a halt would expire the instant it was set."""
    monday = datetime(2025, 6, 16, 9, 0, tzinfo=timezone.utc)
    assert next_utc_monday(monday) == datetime(2025, 6, 23, tzinfo=timezone.utc)
