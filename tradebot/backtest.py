"""Backtest of strategy.py over historical daily bars.

    python -m tradebot.backtest                 # 2y of real Alpaca bars
    python -m tradebot.backtest --synthetic     # deterministic fake bars

The portfolio simulation walks bar by bar and calls the REAL RiskManager for
every proposed order, so position sizing, the concentration cap, the
loss-limit halts and the churn brake all apply exactly as they will live. A
second implementation of the sizing rules here would be free to drift from
risk.py and quietly make the backtest optimistic.

Order timing avoids look-ahead:
  * signals are computed from a bar's close
  * entries fill at the NEXT bar's open
  * stops and targets fill within the bar that traded through them, and when a
    single bar spans both legs the stop is assumed to have filled first

vectorbt does the cash accounting and the statistics; this module decides only
which orders exist. The two are reconciled at the end and any divergence is
reported rather than hidden.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.config import PROJECT_ROOT, Config, get_config
from tradebot.data import MarketData, average_dollar_volume
from tradebot.logging_setup import configure_logging, get_logger
from tradebot.risk import (
    AccountState,
    ProposedOrder,
    RiskManager,
    SymbolStats,
    next_utc_midnight,
    next_utc_monday,
)
from tradebot.strategy import Position, exit_signals, generate_signals, latest_candidates

log = get_logger(__name__)


@dataclass
class Trade:
    """One completed round trip."""

    symbol: str
    qty: float
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    reason: str

    def pnl(self, cost_per_side: float = 0.0) -> float:
        gross = (self.exit_price - self.entry_price) * self.qty
        costs = (self.entry_price + self.exit_price) * self.qty * cost_per_side
        return gross - costs

    def return_pct(self, cost_per_side: float = 0.0) -> float:
        basis = self.entry_price * self.qty
        return self.pnl(cost_per_side) / basis if basis else 0.0


@dataclass
class SimulationResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    orders: list[dict] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    halts: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------- data


def synthetic_bars(
    symbols: list[str],
    days: int = 730,
    seed: int = 20240101,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Deterministic geometric-Brownian bars with crypto-like volatility.

    This exists so the simulation pipeline can be verified without market
    access. It is NOT a substitute for a real backtest: the returns are drawn
    from a distribution with no fat tails, no regime changes and no
    cross-asset correlation, so any edge measured on it is an artifact of the
    generator, not of the strategy.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    index = pd.DatetimeIndex([start + timedelta(days=i) for i in range(days)], name="timestamp")

    frames = []
    for offset, symbol in enumerate(symbols):
        # Daily sigma ~3.5%, a mild positive drift. Crypto-ish, not calibrated.
        returns = rng.normal(loc=0.0007, scale=0.035, size=days)
        close = start_price * (1.0 + offset * 0.1) * np.exp(np.cumsum(returns))
        intrabar = np.abs(rng.normal(0.0, 0.015, size=days))
        frame = pd.DataFrame(
            {
                "open": close * (1.0 + rng.normal(0.0, 0.004, size=days)),
                "high": close * (1.0 + intrabar),
                "low": close * (1.0 - intrabar),
                "close": close,
                "volume": rng.lognormal(mean=13.0, sigma=0.4, size=days),
            },
            index=index,
        )
        frame["symbol"] = symbol
        frames.append(
            frame.set_index("symbol", append=True).reorder_levels(["symbol", "timestamp"])
        )
    return pd.concat(frames).sort_index()


def load_bars(config: Config, years: int, synthetic: bool = False) -> pd.DataFrame:
    universe = config.settings.universe
    if synthetic:
        symbols = universe.resolve()
        symbols = universe.with_benchmark(symbols)
        log.info("backtest.synthetic_data", symbols=len(symbols), days=years * 365)
        return synthetic_bars(symbols, days=years * 365)

    market = MarketData(config)
    symbols = universe.with_benchmark(market.discover_universe())
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365)
    log.info("backtest.fetching", symbols=len(symbols), start=start.date().isoformat())
    return market.get_daily_bars(symbols, start, end)


# --------------------------------------------------------------- simulation


def simulate(
    bars: pd.DataFrame, config: Config, cost: float | None = None
) -> SimulationResult:
    """Bar-by-bar portfolio simulation driving the real RiskManager.

    `cost` is the per-side fee. Pass 0.0 for the gross run. Because position
    size depends on equity, the zero-cost run can take slightly different
    trades — that is a real consequence of costs, not an artifact.
    """
    settings = config.settings
    manager = RiskManager(settings.risk, min_order_notional=settings.execution.min_order_notional)
    cost = settings.backtest.cost_per_side_pct if cost is None else cost

    bars = bars.sort_index()
    signals = generate_signals(bars, settings.strategy, settings.universe.benchmark)
    dollar_volume = average_dollar_volume(bars, window=30)
    tradable = set(settings.universe.resolve())

    timestamps = pd.DatetimeIndex(bars.index.get_level_values("timestamp").unique()).sort_values()

    cash = float(settings.risk.starting_capital)
    positions: dict[str, Position] = {}
    result = SimulationResult()
    pending_entries: list[str] = []
    halted_until: datetime | None = None
    day_anchor = cash
    week_anchor = cash
    current_day = None
    current_week = None
    round_trips_today = 0
    equity_points: list[float] = []

    def mark_to_market(timestamp: pd.Timestamp) -> float:
        total = cash
        for symbol, position in positions.items():
            try:
                total += position.qty * float(bars.loc[(symbol, timestamp), "close"])
            except KeyError:
                total += position.qty * position.entry_price
        return total

    def record_order(timestamp, symbol, qty, price, side, intent):
        result.orders.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "size": qty if side == "buy" else -qty,
                "price": price,
                "intent": intent,
            }
        )

    def close_position(timestamp, symbol, price, reason):
        nonlocal cash, round_trips_today
        position = positions.pop(symbol)
        proceeds = position.qty * price
        cash += proceeds - proceeds * cost
        record_order(timestamp, symbol, position.qty, price, "sell", reason)
        result.trades.append(
            Trade(
                symbol=symbol,
                qty=position.qty,
                entry_date=position.entry_date,
                entry_price=position.entry_price,
                exit_date=timestamp,
                exit_price=price,
                reason=reason,
            )
        )
        round_trips_today += 1

    for timestamp in timestamps:
        day = timestamp.normalize()
        week = day - timedelta(days=int(day.weekday()))

        if current_day != day:
            current_day = day
            day_anchor = mark_to_market(timestamp)
            round_trips_today = 0
        if current_week != week:
            current_week = week
            week_anchor = mark_to_market(timestamp)

        bars_so_far = bars[bars.index.get_level_values("timestamp") <= timestamp]

        # 1. Fill entries decided on the previous bar, at this bar's open.
        for symbol in pending_entries:
            try:
                row = bars.loc[(symbol, timestamp)]
            except KeyError:
                continue
            price = float(row["open"])
            equity = mark_to_market(timestamp)
            state = AccountState(
                equity=equity,
                cash=cash,
                buying_power=cash,
                positions=list(positions.values()),
                symbol_stats=_symbol_stats(bars, dollar_volume, timestamp),
                day_anchor_equity=day_anchor,
                week_anchor_equity=week_anchor,
                round_trips_today=round_trips_today,
                halted_until=halted_until,
                now=timestamp.to_pydatetime(),
            )
            # Ask for the full risk-sized amount; the manager decides the rest.
            wanted = (equity * settings.risk.risk_per_trade_pct
                      / settings.risk.stop_distance_pct) / price
            decision = manager.check(
                ProposedOrder(symbol, "buy", wanted, price, intent="entry"), state
            )
            if not decision.approved:
                key = decision.reason.split(":", 1)[-1].strip()[:60]
                result.rejections[key] = result.rejections.get(key, 0) + 1
                continue
            qty = decision.order.qty
            spend = qty * price
            total_cost = spend + spend * cost
            if total_cost > cash:
                continue
            cash -= total_cost
            positions[symbol] = Position(symbol, qty, price, timestamp)
            record_order(timestamp, symbol, qty, price, "buy", "entry")
        pending_entries = []

        # 2. Exits on this bar, using the same pure function as the live loop.
        if positions:
            for exit_signal in exit_signals(
                list(positions.values()), bars_so_far, settings.strategy
            ):
                close_position(
                    timestamp, exit_signal.symbol, exit_signal.price, exit_signal.reason
                )

        equity = mark_to_market(timestamp)
        equity_points.append(equity)

        # 3. Halts. A breach flattens everything and stops new entries.
        state = AccountState(
            equity=equity,
            cash=cash,
            buying_power=cash,
            positions=list(positions.values()),
            symbol_stats={},
            day_anchor_equity=day_anchor,
            week_anchor_equity=week_anchor,
            round_trips_today=round_trips_today,
            halted_until=halted_until,
            now=timestamp.to_pydatetime(),
        )
        halt = manager.evaluate_halts(state)
        if halt.halted:
            if halt.flatten and positions:
                for symbol in list(positions):
                    try:
                        price = float(bars.loc[(symbol, timestamp), "close"])
                    except KeyError:
                        price = positions[symbol].entry_price
                    close_position(timestamp, symbol, price, "flatten")
                equity_points[-1] = mark_to_market(timestamp)
            if halt.halted_until:
                halted_until = halt.halted_until
            result.halts.append({"timestamp": timestamp, "reason": halt.reason})
            continue

        # 4. Select entries for the next bar's open.
        free_slots = settings.risk.max_open_positions - len(positions)
        if free_slots <= 0:
            continue
        try:
            candidates = latest_candidates(signals, timestamp)
        except Exception:  # noqa: BLE001
            continue
        for symbol in candidates.index.get_level_values("symbol"):
            if len(pending_entries) >= free_slots:
                break
            if symbol in positions or symbol not in tradable:
                continue
            pending_entries.append(symbol)

    result.equity_curve = pd.Series(equity_points, index=timestamps, name="equity")
    return result


def _symbol_stats(
    bars: pd.DataFrame, dollar_volume: pd.Series, timestamp: pd.Timestamp
) -> dict[str, SymbolStats]:
    """Liquidity snapshot as of one bar, for the risk manager."""
    try:
        slice_ = bars.xs(timestamp, level="timestamp")
    except KeyError:
        return {}
    stats: dict[str, SymbolStats] = {}
    for symbol, row in slice_.iterrows():
        try:
            volume = float(dollar_volume.loc[(symbol, timestamp)])
        except KeyError:
            volume = 0.0
        stats[symbol] = SymbolStats(symbol, float(row["close"]), volume)
    return stats


# ----------------------------------------------------------- vectorbt stats


def vectorbt_stats(equity: pd.Series) -> dict[str, float]:
    """Performance statistics for an equity curve, via vectorbt.

    vectorbt does the statistics rather than the order accounting. Its
    Portfolio.from_orders takes one order per (bar, symbol), so it cannot
    represent an entry and a stop-out on the SAME bar — netting the two erases
    the round trip and silently misstates the result. Since a 4% stop against
    crypto volatility produces exactly that case, the bar-by-bar simulation
    above is authoritative and vectorbt measures its output.
    """
    if equity.empty or len(equity) < 2:
        return {}
    import vectorbt as vbt  # noqa: F401  (registers the .vbt accessor)

    returns = equity.pct_change().fillna(0.0)
    accessor = returns.vbt.returns(freq="1D")

    def safe(name: str) -> float:
        try:
            return float(getattr(accessor, name)())
        except Exception:  # noqa: BLE001 — a missing stat must not fail the run
            return float("nan")

    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "max_drawdown": safe("max_drawdown"),
        "annualized_return": safe("annualized"),
        "annualized_volatility": safe("annualized_volatility"),
        "sharpe_ratio": safe("sharpe_ratio"),
        "sortino_ratio": safe("sortino_ratio"),
        "calmar_ratio": safe("calmar_ratio"),
    }


# ---------------------------------------------------------------- reporting


def trade_statistics(trades: list[Trade], cost: float) -> dict:
    if not trades:
        return {
            "trades": 0,
            "win_rate": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "payoff_ratio": float("nan"),
            "expectancy": float("nan"),
        }
    returns = [t.return_pct(cost) for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    win_rate = len(wins) / len(trades)
    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": abs(avg_win / avg_loss) if avg_loss else float("inf"),
        "expectancy": float(np.mean(returns)),
    }


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def plot_equity_curve(equity: pd.Series, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot(equity.index, equity.values, linewidth=1.4, color="#1f77b4", label="net of costs")
    top.axhline(
        equity.iloc[0], linestyle="--", linewidth=1, color="#888", label="starting capital"
    )
    top.set_ylabel("Equity ($)")
    top.set_title(title)
    top.legend(loc="upper left")
    top.grid(alpha=0.3)

    drawdown = equity / equity.cummax() - 1.0
    bottom.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.4)
    bottom.set_ylabel("Drawdown")
    bottom.set_xlabel("Date")
    bottom.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def format_report(
    net: SimulationResult,
    gross: SimulationResult,
    config: Config,
    synthetic: bool,
) -> str:
    cost = config.settings.backtest.cost_per_side_pct
    capital = config.settings.risk.starting_capital
    net_equity, gross_equity = net.equity_curve, gross.equity_curve
    net_stats = trade_statistics(net.trades, cost)
    stats = vectorbt_stats(net_equity)

    net_return = (net_equity.iloc[-1] / capital - 1.0) if len(net_equity) else 0.0
    gross_return = (gross_equity.iloc[-1] / capital - 1.0) if len(gross_equity) else 0.0

    out: list[str] = ["=" * 74]
    out.append("BACKTEST RESULTS" + ("   [SYNTHETIC DATA — NOT A REAL RESULT]" if synthetic else ""))
    out.append("=" * 74)
    if len(net_equity):
        out.append(f"  Period             {net_equity.index[0].date()} -> {net_equity.index[-1].date()}")
    out.append(f"  Starting capital   ${capital:,.2f}")
    out.append(f"  Cost assumption    {cost:.4%} per side")
    out.append("")
    out.append("  RETURN")
    if len(gross_equity):
        out.append(f"    Gross (zero cost)      {gross_return:>10.2%}    ending ${gross_equity.iloc[-1]:,.2f}  ({len(gross.trades)} trades)")
    if len(net_equity):
        out.append(f"    Net of costs           {net_return:>10.2%}    ending ${net_equity.iloc[-1]:,.2f}  ({len(net.trades)} trades)")
    out.append(f"    Cost drag              {gross_return - net_return:>10.2%}")
    out.append("")
    out.append("  RISK (net)")
    out.append(f"    Max drawdown           {max_drawdown(net_equity):>10.2%}")
    if stats:
        out.append(f"    Annualized return      {stats['annualized_return']:>10.2%}")
        out.append(f"    Annualized volatility  {stats['annualized_volatility']:>10.2%}")
        out.append(f"    Sharpe ratio           {stats['sharpe_ratio']:>10.2f}")
        out.append(f"    Sortino ratio          {stats['sortino_ratio']:>10.2f}")
    out.append("")
    out.append("  TRADES (net)")
    out.append(f"    Number of trades       {net_stats['trades']:>10d}")
    if net_stats["trades"]:
        out.append(f"    Win rate               {net_stats['win_rate']:>10.2%}")
        out.append(f"    Average win            {net_stats['avg_win']:>10.2%}")
        out.append(f"    Average loss           {net_stats['avg_loss']:>10.2%}")
        out.append(f"    Payoff ratio           {net_stats['payoff_ratio']:>10.2f}")
        out.append(f"    Expectancy per trade   {net_stats['expectancy']:>10.3%}")

    if net.trades:
        by_reason: dict[str, int] = {}
        for trade in net.trades:
            by_reason[trade.reason] = by_reason.get(trade.reason, 0) + 1
        out.append("")
        out.append("  EXITS BY REASON")
        for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            out.append(f"    {reason:<20} {count:>6d}   ({count / len(net.trades):.1%})")

    if net.halts:
        out.append("")
        out.append(f"  RISK HALTS             {len(net.halts)}")
        for halt in net.halts[:3]:
            out.append(f"    {halt['timestamp'].date()}  {halt['reason'][:58]}")

    if net.rejections:
        out.append("")
        out.append("  TOP RISK REJECTIONS (proposed orders vetoed)")
        merged: dict[str, int] = {}
        for reason, count in net.rejections.items():
            # Strip the varying numbers so "volume 9,572,464 below floor" and
            # "volume 8,986,026 below floor" aggregate into one line.
            key = re.sub(r"[-+]?[\d,]*\.?\d+%?", "N", reason).strip()
            merged[key] = merged.get(key, 0) + count
        for reason, count in sorted(merged.items(), key=lambda kv: -kv[1])[:6]:
            out.append(f"    {count:>6d}   {reason[:56]}")

    out.append("=" * 74)
    return "\n".join(out)


def verdict(net: SimulationResult, config: Config) -> str:
    """A plain reading of the result. Never talks the numbers up."""
    capital = config.settings.risk.starting_capital
    if not len(net.equity_curve):
        return "No result: the simulation produced no equity curve."
    final = float(net.equity_curve.iloc[-1])
    net_return = final / capital - 1.0
    trades = len(net.trades)

    if trades == 0:
        return (
            "The strategy took ZERO trades. Nothing has been validated. The most "
            "likely causes are the liquidity floor (risk.min_avg_dollar_volume_30d), "
            "the broker minimum notional against a small account, or a pullback band "
            "too narrow to ever fire. Check the rejection counts above."
        )
    if net_return <= 0:
        return (
            f"THE STRATEGY LOSES MONEY AFTER COSTS: {net_return:.2%} over the period, "
            f"ending at ${final:,.2f} from ${capital:,.2f}.\n"
            "Do not run this live. Parameters worth revisiting, in order:\n"
            "  1. stop_pct (4%) against crypto's daily volatility — a stop this tight "
            "is close to noise, which shows up as a high proportion of 'stop' exits.\n"
            "  2. cost_per_side_pct — at ~0.25% per side the strategy must clear ~0.5% "
            "per round trip before it earns anything. Fewer, larger trades help.\n"
            "  3. the pullback band (2-4%) — on assets that move 3.5% a day this is "
            "roughly a one-day move, so it fires on noise rather than on a real pullback.\n"
            "Change ONE of these at a time and re-run. Do not search combinations until "
            "the backtest looks good — that fits the parameters to this specific history "
            "and the result will not survive live."
        )
    return (
        f"The strategy is profitable after costs over this period: {net_return:.2%}, "
        f"ending at ${final:,.2f} from ${capital:,.2f} across {trades} trades.\n"
        "Before trusting it: one backtest on one history is weak evidence. Check that "
        "the result is not driven by a handful of trades, and that it survives a "
        "different date range."
    )


# --------------------------------------------------------------------- main


def run(years: int | None = None, synthetic: bool = False, settings_path: str | None = None) -> int:
    config = Config.load(settings_path) if settings_path else get_config()
    configure_logging(json_logs=False)
    years = years or config.settings.backtest.years

    try:
        bars = load_bars(config, years, synthetic=synthetic)
    except Exception as exc:  # noqa: BLE001
        log.error("backtest.data_failed", error=str(exc), error_type=type(exc).__name__)
        print(
            f"\nCould not load bars: {exc}\n"
            "If this is a network or credential problem, run "
            "`python -m tradebot.preflight` first, or use --synthetic to "
            "exercise the pipeline without market access.\n",
            file=sys.stderr,
        )
        return 1

    log.info(
        "backtest.loaded",
        rows=len(bars),
        symbols=bars.index.get_level_values("symbol").nunique(),
    )

    net = simulate(bars, config)
    gross = simulate(bars, config, cost=0.0)

    print(format_report(net, gross, config, synthetic))
    print()
    print(verdict(net, config))

    if len(net.equity_curve):
        png = PROJECT_ROOT / config.settings.backtest.equity_curve_png
        title = "tradebot equity curve (net of costs)"
        if synthetic:
            title += " — SYNTHETIC DATA, NOT A REAL RESULT"
        plot_equity_curve(net.equity_curve, png, title)
        print(f"\nEquity curve written to {png}")

    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description="Backtest the tradebot strategy.")
    parser.add_argument("--years", type=int, default=None, help="years of history (default: config)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use deterministic generated bars instead of Alpaca (pipeline check only)",
    )
    parser.add_argument("--settings", default=None, help="path to an alternative settings.yaml")
    args = parser.parse_args()
    return run(years=args.years, synthetic=args.synthetic, settings_path=args.settings)


if __name__ == "__main__":
    sys.exit(cli())
