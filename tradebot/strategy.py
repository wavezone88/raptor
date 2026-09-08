"""Mean reversion with a trend filter, long only.

Every function here is pure: DataFrame in, DataFrame or list out. No network,
no clock, no broker. That is what lets backtest.py and the live loop share
*identical* logic rather than two implementations that drift apart.

Bar format throughout is Alpaca's: a DataFrame with a MultiIndex of
(symbol, timestamp) and columns open/high/low/close/volume.

No look-ahead: every rolling window ends at the current bar. The volume filter
deliberately compares today's volume against the *prior* N days (shifted by
one) so the average does not contain the value being tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from tradebot.config import StrategySettings, get_config

ExitReason = Literal["stop", "target", "time_stop"]

SIGNAL_COLUMNS = [
    "close",
    "sma",
    "trend_ok",
    "rolling_high",
    "pullback_pct",
    "pullback_ok",
    "avg_volume",
    "volume_ok",
    "rel_strength",
    "entry_signal",
    "entry_rank",
]


@dataclass(frozen=True)
class Position:
    """An open long position, as the bot understands it.

    stop_price and target_price are fixed when the position is opened, not
    recomputed each cycle. Under atr_multiple mode the ATR moves every bar, and
    a stop that drifted with it would let a losing position quietly widen its
    own risk.
    """

    symbol: str
    qty: float
    entry_price: float
    entry_date: pd.Timestamp
    stop_price: float | None = None
    target_price: float | None = None

    def bars_held(self, as_of: pd.Timestamp, trading_days: pd.DatetimeIndex | None = None) -> int:
        """Trading days held. Uses the bar index when given, else calendar days."""
        entry = pd.Timestamp(self.entry_date).tz_localize(None).normalize()
        current = pd.Timestamp(as_of).tz_localize(None).normalize()
        if trading_days is not None and len(trading_days):
            days = pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique().sort_values()
            entry_pos = days.searchsorted(entry, side="left")
            current_pos = days.searchsorted(current, side="left")
            return max(0, int(current_pos - entry_pos))
        return max(0, (current - entry).days)


@dataclass(frozen=True)
class ExitSignal:
    """A decision to close a position, with the reason and the exit price."""

    symbol: str
    reason: ExitReason
    price: float
    qty: float
    entry_price: float

    @property
    def pnl_per_share(self) -> float:
        return self.price - self.entry_price


def _params(params: StrategySettings | None) -> StrategySettings:
    return params if params is not None else get_config().settings.strategy


def _require_bar_columns(bars: pd.DataFrame) -> None:
    missing = {"open", "high", "low", "close", "volume"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")
    if not isinstance(bars.index, pd.MultiIndex) or bars.index.nlevels != 2:
        raise ValueError("bars must have a (symbol, timestamp) MultiIndex")


def average_true_range(bars: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's true range, simple-averaged over `period` bars, per symbol.

    True range is the widest of: today's high-low, and the gap from the previous
    close to today's high or low. It measures how far a symbol actually travels
    in a bar, which is the number a stop should be sized against — a 4% stop
    means something completely different on BTC than on a stablecoin pair.
    """
    high, low, close = bars["high"], bars["low"], bars["close"]
    previous_close = close.groupby(level="symbol").shift(1)

    ranges = pd.concat(
        [
            (high - low).rename("hl"),
            (high - previous_close).abs().rename("hc"),
            (low - previous_close).abs().rename("lc"),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return (
        true_range.groupby(level="symbol")
        .transform(lambda s: s.rolling(period, min_periods=period).mean())
        .rename("atr")
    )


def compute_relative_strength(
    bars: pd.DataFrame,
    benchmark: str,
    lookback: int,
) -> pd.Series:
    """Trailing return minus the benchmark's over the same window.

    Positive means the symbol outperformed the benchmark. Used only for
    ranking, never as an entry condition.
    """
    close = bars["close"]
    own_return = close.groupby(level="symbol").transform(lambda s: s / s.shift(lookback) - 1.0)

    symbols = bars.index.get_level_values("symbol")
    if benchmark not in set(symbols):
        # Without the benchmark we can still rank, just on absolute return.
        return own_return.rename("rel_strength")

    benchmark_close = bars.xs(benchmark, level="symbol")["close"].sort_index()
    benchmark_return = benchmark_close / benchmark_close.shift(lookback) - 1.0
    timestamps = bars.index.get_level_values("timestamp")
    aligned = pd.Series(
        benchmark_return.reindex(timestamps).to_numpy(),
        index=bars.index,
        name="benchmark_return",
    )
    return (own_return - aligned).rename("rel_strength")


def generate_signals(
    bars: pd.DataFrame,
    params: StrategySettings | None = None,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Entry signals and their ranking for every (symbol, timestamp) in `bars`.

    Entry requires all three, on the same bar:
      1. trend  — close above the N-day simple moving average
      2. pullback — close is 2%..4% below the rolling N-day high
      3. volume — today's volume above the prior N-day average

    The returned frame is indexed exactly like `bars`, so the live loop can take
    the last timestamp and the backtester can use the whole matrix.
    `entry_rank` is 1 for the strongest candidate on each bar (highest
    3-month return relative to the benchmark), NaN where there is no signal.
    """
    settings = _params(params)
    _require_bar_columns(bars)
    if benchmark is None:
        benchmark = get_config().settings.universe.benchmark

    bars = bars.sort_index()
    grouped_close = bars["close"].groupby(level="symbol")
    grouped_high = bars["high"].groupby(level="symbol")
    grouped_volume = bars["volume"].groupby(level="symbol")

    sma = grouped_close.transform(
        lambda s: s.rolling(settings.trend_sma_days, min_periods=settings.trend_sma_days).mean()
    )
    rolling_high = grouped_high.transform(
        lambda s: s.rolling(
            settings.pullback_lookback_days, min_periods=settings.pullback_lookback_days
        ).max()
    )
    # Prior-N-day average, shifted so today's volume is not part of its own test.
    avg_volume = grouped_volume.transform(
        lambda s: s.shift(1).rolling(
            settings.volume_avg_days, min_periods=settings.volume_avg_days
        ).mean()
    )

    close = bars["close"]
    pullback_pct = (rolling_high - close) / rolling_high

    trend_ok = close > sma
    pullback_ok = (pullback_pct >= settings.pullback_min_pct) & (
        pullback_pct <= settings.pullback_max_pct
    )
    volume_ok = bars["volume"] > (avg_volume * settings.volume_multiple)

    entry_signal = (
        trend_ok.fillna(False) & pullback_ok.fillna(False) & volume_ok.fillna(False)
    )

    rel_strength = compute_relative_strength(
        bars, benchmark, settings.rel_strength_lookback_days
    )

    signals = pd.DataFrame(
        {
            "close": close,
            "sma": sma,
            "trend_ok": trend_ok.fillna(False),
            "rolling_high": rolling_high,
            "pullback_pct": pullback_pct,
            "pullback_ok": pullback_ok.fillna(False),
            "avg_volume": avg_volume,
            "volume_ok": volume_ok.fillna(False),
            "rel_strength": rel_strength,
            "entry_signal": entry_signal,
        },
        index=bars.index,
    )
    signals["entry_rank"] = _rank_signals(signals)
    return signals[SIGNAL_COLUMNS]


def _rank_signals(signals: pd.DataFrame) -> pd.Series:
    """Rank 1..N within each timestamp, best relative strength first."""
    ranks = pd.Series(np.nan, index=signals.index, dtype="float64")
    candidates = signals["entry_signal"]
    if not candidates.any():
        return ranks
    # NaN relative strength (insufficient history) ranks last rather than
    # dropping the candidate entirely.
    scores = signals.loc[candidates, "rel_strength"].fillna(-np.inf)
    ranks.loc[candidates] = (
        scores.groupby(level="timestamp").rank(ascending=False, method="first")
    )
    return ranks


def latest_candidates(
    signals: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Signalling symbols on the most recent bar, best-ranked first.

    This is the bridge the live loop uses; the backtester consumes the full
    `signals` matrix instead. Both read the same columns.
    """
    if signals.empty:
        return signals
    timestamps = signals.index.get_level_values("timestamp")
    target = pd.Timestamp(as_of) if as_of is not None else timestamps.max()
    latest = signals[timestamps == target]
    fired = latest[latest["entry_signal"]]
    return fired.sort_values("entry_rank")


def exit_signals(
    positions: Iterable[Position],
    bars: pd.DataFrame,
    params: StrategySettings | None = None,
) -> list[ExitSignal]:
    """Which open positions must be closed on the latest bar, and why.

    Checked in order of severity: stop, then target, then time stop. If a
    single bar trades through both the stop and the target we assume the stop
    filled — the pessimistic assumption, so the backtest cannot flatter itself.
    """
    settings = _params(params)
    _require_bar_columns(bars)

    exits: list[ExitSignal] = []
    if bars.empty:
        return exits

    bars = bars.sort_index()
    trading_days = pd.DatetimeIndex(bars.index.get_level_values("timestamp").unique())

    for position in positions:
        try:
            symbol_bars = bars.xs(position.symbol, level="symbol")
        except KeyError:
            # No data for a symbol we hold. Never guess an exit price; the
            # caller's staleness check is what protects us here.
            continue
        if symbol_bars.empty:
            continue

        bar = symbol_bars.iloc[-1]
        as_of = symbol_bars.index[-1]

        # Levels fixed at entry win; fall back to computing them for a position
        # adopted by reconciliation, which has no stored levels.
        if position.stop_price is not None and position.target_price is not None:
            stop_price, target_price = position.stop_price, position.target_price
        else:
            stop_price, target_price = stop_and_target_prices(
                position.entry_price, settings
            )

        if float(bar["low"]) <= stop_price:
            exits.append(
                ExitSignal(position.symbol, "stop", stop_price, position.qty, position.entry_price)
            )
            continue
        if float(bar["high"]) >= target_price:
            exits.append(
                ExitSignal(
                    position.symbol, "target", target_price, position.qty, position.entry_price
                )
            )
            continue
        if position.bars_held(as_of, trading_days) >= settings.time_stop_days:
            exits.append(
                ExitSignal(
                    position.symbol,
                    "time_stop",
                    float(bar["close"]),
                    position.qty,
                    position.entry_price,
                )
            )

    return exits


def stop_and_target_prices(
    entry_price: float,
    params: StrategySettings | None = None,
    atr: float | None = None,
) -> tuple[float, float]:
    """Stop and target for an entry. Shared by execution.py and the backtest.

    Under atr_multiple mode, `atr` is the symbol's ATR at entry. If it is
    missing or unusable — too little history, a bad bar — this falls back to
    the fixed percentage rather than returning no stop. Always having a stop
    matters more than always having the preferred one.
    """
    settings = _params(params)
    if settings.stop_mode == "atr_multiple" and atr is not None and atr > 0 and np.isfinite(atr):
        stop = entry_price - settings.atr_stop_multiple * atr
        target = entry_price + settings.atr_target_multiple * atr
        # An ATR wider than the entry price would imply a negative stop.
        if stop > 0:
            return stop, target
    return (
        entry_price * (1.0 - settings.stop_pct),
        entry_price * (1.0 + settings.target_pct),
    )


def stop_distance_fraction(entry_price: float, stop_price: float) -> float:
    """Stop distance as a fraction of entry — what position sizing divides by."""
    if entry_price <= 0:
        return 0.0
    return max(0.0, (entry_price - stop_price) / entry_price)


def is_adding_to_loser(
    symbol: str, last_price: float, positions: Iterable[Position]
) -> bool:
    """True if we already hold `symbol` and it is underwater.

    'Never add to a losing position.' The binding enforcement is in
    RiskManager.check(); this helper exists so the backtester applies the same
    rule without importing the risk manager.
    """
    for position in positions:
        if position.symbol == symbol and last_price < position.entry_price:
            return True
    return False
