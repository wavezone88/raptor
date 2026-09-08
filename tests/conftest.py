"""Shared synthetic-data builders.

Tests construct bars by hand rather than recording fixtures from Alpaca, so
every rule can be exercised at its exact boundary and nothing depends on the
network or on a particular day's market history.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tradebot.config import (
    AlertSettings,
    BacktestSettings,
    DataSettings,
    ExecutionSettings,
    RiskSettings,
    ScheduleSettings,
    StateSettings,
    StrategySettings,
    UniverseSettings,
)

START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def bars_for_symbol(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
    volumes: list[float] | None = None,
    start: datetime = START,
) -> pd.DataFrame:
    """One symbol's daily bars, indexed by timestamp."""
    count = len(closes)
    index = pd.DatetimeIndex(
        [start + timedelta(days=i) for i in range(count)], name="timestamp"
    )
    return pd.DataFrame(
        {
            "open": opens or list(closes),
            "high": highs or [c * 1.001 for c in closes],
            "low": lows or [c * 0.999 for c in closes],
            "close": list(closes),
            "volume": volumes or [1_000.0] * count,
        },
        index=index,
    )


def stack(**by_symbol: pd.DataFrame) -> pd.DataFrame:
    """Combine per-symbol frames into the (symbol, timestamp) MultiIndex shape.

    Keyword names use '_' for '/', so BTC_USD becomes BTC/USD.
    """
    frames = []
    for name, frame in by_symbol.items():
        symbol = name.replace("_", "/")
        copy = frame.copy()
        copy["symbol"] = symbol
        frames.append(copy.set_index("symbol", append=True).reorder_levels(["symbol", "timestamp"]))
    return pd.concat(frames).sort_index()


def uptrend(
    length: int = 30,
    start_price: float = 100.0,
    step: float = 1.0,
    volume: float = 1_000.0,
) -> dict[str, list[float]]:
    """A clean rising series: close is always above any trailing SMA."""
    closes = [start_price + i * step for i in range(length)]
    return {
        "closes": closes,
        "highs": [c * 1.001 for c in closes],
        "lows": [c * 0.999 for c in closes],
        "volumes": [volume] * length,
    }


def signalling_bars(
    pullback_pct: float = 0.03,
    volume_multiple: float = 2.0,
    trend_ok: bool = True,
    length: int = 30,
) -> pd.DataFrame:
    """Bars engineered so the final bar fires (or deliberately fails) a signal.

    The last bar's close sits `pullback_pct` below a spike in the preceding
    bar's high, and its volume is `volume_multiple` times the baseline.
    """
    series = uptrend(length=length)
    closes, highs, lows, volumes = (
        series["closes"],
        series["highs"],
        series["lows"],
        series["volumes"],
    )

    if not trend_ok:
        # Collapse the last close far below the moving average.
        closes[-1] = closes[0] * 0.5
        highs[-1] = closes[-1] * 1.001
        lows[-1] = closes[-1] * 0.999

    last_close = closes[-1]
    # Put the rolling-window high one bar back, at the level that makes the
    # final close exactly `pullback_pct` below it.
    if pullback_pct > 0:
        highs[-2] = last_close / (1.0 - pullback_pct)
    volumes[-1] = volumes[0] * volume_multiple

    return bars_for_symbol(closes, highs=highs, lows=lows, volumes=volumes)


@pytest.fixture
def params() -> StrategySettings:
    """Small windows so tests stay short and readable."""
    return StrategySettings(
        trend_sma_days=5,
        pullback_lookback_days=3,
        pullback_min_pct=0.02,
        pullback_max_pct=0.04,
        volume_avg_days=3,
        volume_multiple=1.0,
        rel_strength_lookback_days=5,
        stop_pct=0.04,
        target_pct=0.08,
        time_stop_days=5,
    )


@pytest.fixture
def risk_settings() -> RiskSettings:
    return RiskSettings(
        starting_capital=50.0,
        max_open_positions=3,
        max_position_pct_equity=0.40,
        risk_per_trade_pct=0.01,
        stop_distance_pct=0.04,
        daily_loss_limit_pct=0.05,
        weekly_loss_limit_pct=0.10,
        session_anchor_timezone="UTC",
        max_round_trips_per_day=3,
        min_avg_dollar_volume_30d=10_000_000.0,
        min_price=0.0,
        fractional_qty_decimals=8,
    )


@pytest.fixture
def execution_settings() -> ExecutionSettings:
    return ExecutionSettings(
        entry_limit_buffer_pct=0.001,
        cancel_unfilled_after_cycles=2,
        stop_policy="exchange_stop_limit",
        stop_limit_slippage_pct=0.005,
        min_order_notional=1.0,
    )


@pytest.fixture
def universe_settings() -> UniverseSettings:
    return UniverseSettings(
        symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
        discover_from_broker=False,
        exclude=["USDT/USD"],
        benchmark="BTC/USD",
    )


@pytest.fixture
def schedule_settings() -> ScheduleSettings:
    return ScheduleSettings(interval_minutes=15, timezone="UTC", heartbeat_hours_utc=[0, 12])


@pytest.fixture
def data_settings() -> DataSettings:
    return DataSettings(
        cache_dir="data_cache",
        bar_timeframe="1Day",
        stale_quote_interval_multiple=2.0,
        max_bar_age_calendar_days=2,
    )


@pytest.fixture
def alert_settings() -> AlertSettings:
    return AlertSettings(channel="discord", dedupe_window_seconds=300)


@pytest.fixture
def backtest_settings() -> BacktestSettings:
    return BacktestSettings(years=2, cost_per_side_pct=0.0025, equity_curve_png="curve.png")


@pytest.fixture
def state_settings() -> StateSettings:
    return StateSettings(dir="state", file="bot_state.json")
