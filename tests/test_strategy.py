"""Tests for strategy.py.

Each entry condition is tested at its boundary and with the condition
deliberately broken, so removing a filter from generate_signals() makes a test
fail rather than silently widening the strategy.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from tradebot.strategy import (
    ExitSignal,
    Position,
    exit_signals,
    generate_signals,
    is_adding_to_loser,
    latest_candidates,
    stop_and_target_prices,
)

from .conftest import START, bars_for_symbol, signalling_bars, stack, uptrend

BENCHMARK = "BTC/USD"


def last_signal(bars, params, symbol="ETH/USD", benchmark=BENCHMARK):
    signals = generate_signals(stack(**{symbol.replace("/", "_"): bars}), params, benchmark)
    return signals.xs(symbol, level="symbol").iloc[-1]


# ------------------------------------------------------------------ entries


def test_all_conditions_met_fires_signal(params):
    row = last_signal(signalling_bars(pullback_pct=0.03, volume_multiple=2.0), params)
    assert row["trend_ok"]
    assert row["pullback_ok"]
    assert row["volume_ok"]
    assert row["entry_signal"]


def test_trend_filter_blocks_signal_below_sma(params):
    """Close under the moving average must veto, however good the pullback."""
    row = last_signal(signalling_bars(trend_ok=False, volume_multiple=2.0), params)
    assert not row["trend_ok"]
    assert not row["entry_signal"]


@pytest.mark.parametrize("pullback", [0.005, 0.015])
def test_shallow_pullback_is_rejected(params, pullback):
    """Under 2% off the rolling high is not a pullback we trade."""
    row = last_signal(signalling_bars(pullback_pct=pullback, volume_multiple=2.0), params)
    assert row["pullback_pct"] < params.pullback_min_pct
    assert not row["pullback_ok"]
    assert not row["entry_signal"]


@pytest.mark.parametrize("pullback", [0.06, 0.12])
def test_deep_pullback_is_rejected(params, pullback):
    """Past 4% we assume the trend is breaking, not mean-reverting."""
    row = last_signal(signalling_bars(pullback_pct=pullback, volume_multiple=2.0), params)
    assert row["pullback_pct"] > params.pullback_max_pct
    assert not row["pullback_ok"]
    assert not row["entry_signal"]


def band_edge_bars(last_close: float) -> pd.DataFrame:
    """Bars whose final pullback is exactly (rolling_high - last_close) / 100.

    Built against a rolling high of exactly 100.0 so the division is exact in
    binary and the boundary comparison is not decided by a rounding error.
    """
    closes = [88.0] * 20 + [92.0, 93.0, 94.0, 95.0, last_close]
    highs = [c * 1.001 for c in closes]
    highs[-2] = 100.0          # the rolling-window high
    highs[-1] = last_close      # today's high must not exceed it
    volumes = [1_000.0] * len(closes)
    volumes[-1] = 2_000.0
    return bars_for_symbol(closes, highs=highs, volumes=volumes)


@pytest.mark.parametrize(
    ("last_close", "expected_pullback"), [(98.0, 0.02), (96.0, 0.04)]
)
def test_pullback_band_is_inclusive_at_both_edges(params, last_close, expected_pullback):
    """2% and 4% are inside the band, not outside it."""
    row = last_signal(band_edge_bars(last_close), params)
    assert row["pullback_pct"] == expected_pullback
    assert row["pullback_ok"]
    assert row["entry_signal"]


@pytest.mark.parametrize("last_close", [98.1, 95.9])
def test_pullback_just_outside_the_band_is_rejected(params, last_close):
    """1.9% and 4.1% fall outside; the band edges are the only accepted values
    at the boundary, so widening or narrowing the band breaks a test."""
    row = last_signal(band_edge_bars(last_close), params)
    assert not row["pullback_ok"]
    assert not row["entry_signal"]


def test_low_volume_blocks_signal(params):
    """Volume at or below the trailing average must veto."""
    row = last_signal(signalling_bars(pullback_pct=0.03, volume_multiple=0.5), params)
    assert not row["volume_ok"]
    assert not row["entry_signal"]


def test_volume_average_excludes_the_bar_being_tested(params):
    """No look-ahead: the average is the PRIOR N days, so a volume spike
    cannot inflate the threshold it is being compared against."""
    bars = signalling_bars(pullback_pct=0.03, volume_multiple=10.0)
    row = last_signal(bars, params)
    baseline = bars["volume"].iloc[0]
    assert row["avg_volume"] == pytest.approx(baseline)
    assert row["volume_ok"]


def test_insufficient_history_produces_no_signal(params):
    """Before the SMA window fills there is nothing to trade on."""
    short = bars_for_symbol([100.0, 101.0, 102.0])
    signals = generate_signals(stack(ETH_USD=short), params, BENCHMARK)
    assert not signals["entry_signal"].any()


def test_signals_are_indexed_like_the_input(params):
    bars = stack(ETH_USD=signalling_bars(), SOL_USD=signalling_bars())
    signals = generate_signals(bars, params, BENCHMARK)
    assert signals.index.equals(bars.index)


def test_generate_signals_does_not_mutate_input(params):
    """Pure function: the backtester reuses the same frame across runs."""
    bars = stack(ETH_USD=signalling_bars())
    before = bars.copy(deep=True)
    generate_signals(bars, params, BENCHMARK)
    pd.testing.assert_frame_equal(bars, before)


def test_missing_columns_raise(params):
    bars = stack(ETH_USD=signalling_bars()).drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        generate_signals(bars, params, BENCHMARK)


# ------------------------------------------------------------------ ranking


def test_ranking_prefers_strongest_relative_to_benchmark(params):
    """Two identical setups rank by return relative to the benchmark."""
    flat = uptrend(length=30, step=0.0)
    benchmark = bars_for_symbol(flat["closes"], volumes=flat["volumes"])

    strong = signalling_bars(pullback_pct=0.03, volume_multiple=2.0)          # rising
    weak_series = uptrend(length=30, step=0.1)                                # barely rising
    weak = signalling_bars(pullback_pct=0.03, volume_multiple=2.0)
    weak["close"] = weak_series["closes"]
    weak["high"] = [c * 1.001 for c in weak_series["closes"]]
    weak["low"] = [c * 0.999 for c in weak_series["closes"]]
    weak.iloc[-2, weak.columns.get_loc("high")] = weak["close"].iloc[-1] / (1 - 0.03)
    weak.iloc[-1, weak.columns.get_loc("volume")] = 2000.0

    signals = generate_signals(
        stack(BTC_USD=benchmark, ETH_USD=strong, SOL_USD=weak), params, BENCHMARK
    )
    fired = latest_candidates(signals)
    ranked = list(fired.index.get_level_values("symbol"))
    assert ranked[0] == "ETH/USD", "stronger relative strength must rank first"
    assert fired.loc[("ETH/USD", fired.index[0][1]), "entry_rank"] == 1.0


def test_relative_strength_is_measured_against_the_benchmark(params):
    """A symbol tracking the benchmark exactly has ~zero relative strength."""
    series = uptrend(length=30)
    identical = bars_for_symbol(series["closes"], volumes=series["volumes"])
    signals = generate_signals(
        stack(BTC_USD=identical, ETH_USD=identical.copy()), params, BENCHMARK
    )
    assert signals.xs("ETH/USD", level="symbol")["rel_strength"].iloc[-1] == pytest.approx(0.0)


def test_ranks_are_unique_within_a_timestamp(params):
    bars = stack(
        ETH_USD=signalling_bars(), SOL_USD=signalling_bars(), AVAX_USD=signalling_bars()
    )
    signals = generate_signals(bars, params, BENCHMARK)
    fired = latest_candidates(signals)
    ranks = fired["entry_rank"].tolist()
    assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_latest_candidates_returns_only_signalling_symbols(params):
    bars = stack(ETH_USD=signalling_bars(), SOL_USD=signalling_bars(volume_multiple=0.5))
    fired = latest_candidates(generate_signals(bars, params, BENCHMARK))
    assert list(fired.index.get_level_values("symbol")) == ["ETH/USD"]


# -------------------------------------------------------------------- exits


def position(symbol="ETH/USD", entry_price=100.0, qty=1.0, entry_offset_days=0):
    return Position(
        symbol=symbol,
        qty=qty,
        entry_price=entry_price,
        entry_date=START + timedelta(days=entry_offset_days),
    )


def test_stop_exit_fires_at_minus_four_percent(params):
    """Low touching -4% closes the position at the stop price."""
    bars = stack(ETH_USD=bars_for_symbol([100.0] * 4 + [96.0], lows=[100.0] * 4 + [95.9]))
    exits = exit_signals([position(entry_price=100.0)], bars, params)
    assert len(exits) == 1
    assert exits[0].reason == "stop"
    assert exits[0].price == pytest.approx(96.0)


def test_no_stop_just_above_the_threshold(params):
    """-3.9% must not trigger the -4% stop."""
    bars = stack(ETH_USD=bars_for_symbol([100.0] * 4 + [96.1], lows=[100.0] * 4 + [96.1]))
    assert exit_signals([position(entry_price=100.0)], bars, params) == []


def test_target_exit_fires_at_plus_eight_percent(params):
    bars = stack(
        ETH_USD=bars_for_symbol([100.0] * 4 + [108.5], highs=[100.0] * 4 + [108.5])
    )
    exits = exit_signals([position(entry_price=100.0)], bars, params)
    assert len(exits) == 1
    assert exits[0].reason == "target"
    assert exits[0].price == pytest.approx(108.0)


def test_stop_wins_when_one_bar_hits_both_legs(params):
    """Pessimistic assumption — the backtest must not flatter itself by
    assuming the target filled first."""
    bars = stack(
        ETH_USD=bars_for_symbol(
            [100.0] * 4 + [104.0], highs=[100.0] * 4 + [109.0], lows=[100.0] * 4 + [95.0]
        )
    )
    exits = exit_signals([position(entry_price=100.0)], bars, params)
    assert [e.reason for e in exits] == ["stop"]


def test_time_stop_fires_after_five_days(params):
    """Held five bars with neither leg hit — close it and free the slot."""
    bars = stack(ETH_USD=bars_for_symbol([100.0] * 6))
    exits = exit_signals([position(entry_price=100.0, entry_offset_days=0)], bars, params)
    assert len(exits) == 1
    assert exits[0].reason == "time_stop"
    assert exits[0].price == pytest.approx(100.0)


def test_time_stop_does_not_fire_early(params):
    bars = stack(ETH_USD=bars_for_symbol([100.0] * 5))
    assert exit_signals([position(entry_price=100.0, entry_offset_days=0)], bars, params) == []


def test_no_exit_when_position_is_quietly_in_profit(params):
    bars = stack(ETH_USD=bars_for_symbol([100.0, 101.0, 102.0]))
    assert exit_signals([position(entry_price=100.0)], bars, params) == []


def test_missing_bars_never_invent_an_exit(params):
    """A symbol with no data is skipped, not exited at a guessed price."""
    bars = stack(SOL_USD=bars_for_symbol([100.0] * 10))
    assert exit_signals([position(symbol="ETH/USD")], bars, params) == []


def test_exit_carries_qty_and_entry_for_pnl(params):
    bars = stack(ETH_USD=bars_for_symbol([100.0] * 4 + [96.0], lows=[100.0] * 4 + [95.0]))
    exit_signal = exit_signals([position(entry_price=100.0, qty=2.5)], bars, params)[0]
    assert exit_signal.qty == 2.5
    assert exit_signal.entry_price == 100.0
    assert exit_signal.pnl_per_share == pytest.approx(-4.0)


def test_multiple_positions_are_evaluated_independently(params):
    bars = stack(
        ETH_USD=bars_for_symbol([100.0] * 4 + [96.0], lows=[100.0] * 4 + [95.0]),
        SOL_USD=bars_for_symbol([100.0] * 4 + [108.5], highs=[100.0] * 4 + [109.0]),
    )
    exits = exit_signals(
        [position(symbol="ETH/USD"), position(symbol="SOL/USD")], bars, params
    )
    assert {e.symbol: e.reason for e in exits} == {"ETH/USD": "stop", "SOL/USD": "target"}


# ------------------------------------------------------------------ helpers


def test_stop_and_target_prices(params):
    stop, target = stop_and_target_prices(100.0, params)
    assert stop == pytest.approx(96.0)
    assert target == pytest.approx(108.0)


def test_never_add_to_a_losing_position():
    held = [Position("ETH/USD", 1.0, 100.0, START)]
    assert is_adding_to_loser("ETH/USD", 95.0, held) is True
    assert is_adding_to_loser("ETH/USD", 105.0, held) is False
    assert is_adding_to_loser("SOL/USD", 1.0, held) is False
