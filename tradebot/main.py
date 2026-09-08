"""Scheduler and the one run_cycle() that does all the work.

    python -m tradebot.main                # run on the schedule, forever
    python -m tradebot.main --once         # a single cycle, then exit
    python -m tradebot.main --flatten      # close everything and exit
    python -m tradebot.main --offline-demo # a full cycle against a fake broker

Cycle order, which is deliberate:

    kill switch -> reconcile -> halts -> exits -> stops -> stale orders
                -> scan entries -> risk -> execute -> persist -> summary

Exits are managed before entries so a freed slot is usable in the same cycle
and, more importantly, so risk is reduced before it is added. Reconciliation
runs first because every later decision depends on knowing what we actually
hold.

The whole cycle is wrapped so that no exception can kill the process: a bot
that dies with positions open is worse than one that skips a cycle. Under the
bot_managed stop policy it is much worse, because nothing at the exchange is
protecting the position.
"""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from tradebot.alerts import Alert, Alerter, Severity
from tradebot.config import Config, get_config
from tradebot.data import DataError, MarketData, average_dollar_volume
from tradebot.execution import AlpacaBroker, ExecutionEngine, ExecutionError
from tradebot.logging_setup import configure_logging, get_logger
from tradebot.risk import AccountState, ProposedOrder, RiskManager, SymbolStats
from tradebot.state import BotState
from tradebot.strategy import (
    average_true_range,
    exit_signals,
    generate_signals,
    latest_candidates,
    stop_and_target_prices,
    stop_distance_fraction,
)

log = get_logger(__name__)


class TradingBot:
    def __init__(
        self,
        config: Config | None = None,
        market: Any | None = None,
        broker: Any | None = None,
        alerter: Alerter | None = None,
    ):
        self.config = config or get_config()
        self.market = market or MarketData(self.config)
        self.broker = broker or AlpacaBroker(self.config)
        self.execution = ExecutionEngine(self.broker, self.config)
        self.risk = RiskManager(
            self.config.settings.risk,
            min_order_notional=self.config.settings.execution.min_order_notional,
        )
        self.alerter = alerter or Alerter(self.config)
        self.state = BotState.load(self.config.settings.state_path)
        self._last_heartbeat_hour: int | None = None
        self._shutdown = False

    # ------------------------------------------------------------ kill switch

    def kill_switch_active(self) -> bool:
        return self.config.settings.stop_file_path.exists()

    def flatten_and_exit(self, reason: str) -> int:
        """Cancel everything, close everything, alert, persist, exit."""
        log.warning("bot.flatten_requested", reason=reason)
        try:
            cancelled = self.execution.cancel_all_orders(self.state)
            flattened = self.execution.flatten_all(self.state, self.state.cycle)
            self.alerter.kill_switch(len(flattened), cancelled)
            log.warning(
                "bot.flattened", positions=len(flattened), orders_cancelled=cancelled
            )
        except Exception as exc:  # noqa: BLE001
            log.error("bot.flatten_failed", error=str(exc), error_type=type(exc).__name__)
            self.alerter.exception("flatten_and_exit", exc)
            self._persist()
            return 1
        self._persist()
        return 0

    def _persist(self) -> None:
        try:
            self.state.save(self.config.settings.state_path)
        except Exception as exc:  # noqa: BLE001
            log.error("state.save_failed", error=str(exc))

    # ------------------------------------------------------------------ cycle

    def run_cycle(self) -> dict:
        """One full cycle. Never raises — every failure is alerted and logged."""
        summary: dict[str, Any] = {"actions": [], "rejections": [], "errors": []}
        now = datetime.now(timezone.utc)

        try:
            if self.kill_switch_active():
                log.warning("bot.kill_switch_detected", file=str(self.config.settings.stop_file_path))
                self.flatten_and_exit("STOP file present")
                self._shutdown = True
                summary["actions"].append("kill_switch")
                return summary

            self.state.touch(now)
            cycle = self.state.cycle
            log.info("cycle.start", cycle=cycle, at=now.isoformat())

            # 1. Reconcile. Alpaca is the source of truth.
            report = self.execution.reconcile(self.state)
            if report.has_mismatch:
                self.alerter.reconciliation_mismatch(report.describe())
                summary["actions"].append(f"reconciliation: {report.describe()}")

            # 2. Account snapshot and loss anchors.
            account = self.broker.get_account()
            rolled = self.state.roll_anchors(now, account.equity)
            if rolled:
                log.info("cycle.anchors_rolled", **rolled)

            # 3. Halts. A breach flattens and stops.
            account_state = self._account_state(account, {}, now)
            halt = self.risk.evaluate_halts(account_state)
            if halt.halted:
                if halt.flatten:
                    flattened = self.execution.flatten_all(self.state, cycle)
                    self.state.set_halt(halt.halted_until, halt.reason)
                    self.alerter.halt(halt.reason, halt.halted_until, len(flattened))
                    summary["actions"].append(f"HALT+FLATTEN: {halt.reason}")
                else:
                    log.info("cycle.halted", reason=halt.reason)
                    summary["actions"].append(f"halted: {halt.reason}")
                self._persist()
                self._log_summary(cycle, account, summary)
                return summary
            if self.state.halted_until and not self.state.is_halted(now):
                log.info("cycle.halt_expired", was=self.state.halt_reason)
                self.state.clear_halt()

            # 4. Market data. Any staleness or failure means no action at all.
            universe = self.market.discover_universe()
            symbols = self.config.settings.universe.with_benchmark(universe)
            start = now - timedelta(days=self._history_days())
            try:
                bars = self.market.get_daily_bars(symbols, start, now)
                held = list(self.state.positions)
                self.market.assert_bars_fresh(bars, held or None, now)
            except DataError as exc:
                log.warning("cycle.no_action_stale_data", reason=str(exc))
                summary["errors"].append(f"data: {exc}")
                self._persist()
                self._log_summary(cycle, account, summary)
                return summary

            dollar_volume = average_dollar_volume(bars, window=30)
            stats = self._symbol_stats(bars, dollar_volume)

            # 5. Exits before entries: reduce risk before adding it.
            for exit_signal in exit_signals(
                self.state.as_positions(), bars, self.config.settings.strategy
            ):
                decision = self.risk.check(
                    ProposedOrder(
                        exit_signal.symbol,
                        "sell",
                        exit_signal.qty,
                        exit_signal.price,
                        intent=exit_signal.reason,
                    ),
                    self._account_state(account, stats, now),
                )
                if not decision.approved:
                    log.error("cycle.exit_rejected", symbol=exit_signal.symbol, reason=decision.reason)
                    summary["errors"].append(f"exit rejected: {decision.reason}")
                    continue
                try:
                    self.execution.place_exit(
                        exit_signal.symbol, decision.order.qty, exit_signal.reason, self.state, cycle
                    )
                    pnl = self.state.record_exit(exit_signal.symbol, exit_signal.price, now)
                    self.alerter.trade(
                        exit_signal.symbol, "sell", decision.order.qty,
                        exit_signal.price, exit_signal.reason, pnl,
                    )
                    summary["actions"].append(
                        f"EXIT {exit_signal.symbol} ({exit_signal.reason}) pnl={pnl:.2f}"
                    )
                except ExecutionError as exc:
                    log.error("cycle.exit_failed", symbol=exit_signal.symbol, error=str(exc))
                    summary["errors"].append(f"exit failed: {exc}")

            # 6. Every open position must carry a protective stop.
            self._ensure_protective_stops(cycle, summary)

            # 7. Stale entry limits: a stale price is a stale trade.
            for symbol in self.execution.cancel_stale_entries(self.state, cycle):
                summary["actions"].append(f"cancelled stale entry {symbol}")

            # 8. Scan for entries.
            self._scan_entries(bars, stats, account, now, cycle, summary)

            self._persist()
            self._maybe_heartbeat(account, now)
            self._log_summary(cycle, account, summary)
            return summary

        except Exception as exc:  # noqa: BLE001 — the process must never die here
            log.error(
                "cycle.unhandled_exception",
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,
            )
            self.alerter.exception("run_cycle", exc)
            summary["errors"].append(f"{type(exc).__name__}: {exc}")
            self._persist()
            return summary

    # ---------------------------------------------------------------- helpers

    def _history_days(self) -> int:
        """Enough history for the longest indicator window, plus slack."""
        strategy = self.config.settings.strategy
        longest = max(
            strategy.trend_sma_days,
            strategy.volume_avg_days,
            strategy.rel_strength_lookback_days,
            strategy.pullback_lookback_days,
        )
        return int(longest * 2 + 30)

    def _account_state(self, account, stats: dict, now: datetime) -> AccountState:
        return AccountState(
            equity=account.equity,
            cash=account.cash,
            buying_power=account.buying_power,
            positions=self.state.as_positions(),
            symbol_stats=stats,
            day_anchor_equity=self.state.day_anchor_equity,
            week_anchor_equity=self.state.week_anchor_equity,
            round_trips_today=self.state.round_trips_today,
            halted_until=self.state.halted_until_dt,
            now=now,
        )

    @staticmethod
    def _symbol_stats(bars: pd.DataFrame, dollar_volume: pd.Series) -> dict[str, SymbolStats]:
        stats: dict[str, SymbolStats] = {}
        if bars.empty:
            return stats
        timestamps = bars.index.get_level_values("timestamp")
        latest = timestamps.max()
        for symbol in dict.fromkeys(bars.index.get_level_values("symbol")):
            try:
                row = bars.loc[(symbol, latest)]
                volume = float(dollar_volume.loc[(symbol, latest)])
            except KeyError:
                continue
            stats[symbol] = SymbolStats(symbol, float(row["close"]), volume)
        return stats

    def _ensure_protective_stops(self, cycle: int, summary: dict) -> None:
        """Place a resting stop for any position that lacks one.

        Runs every cycle, so a stop is restored after a restart or after a
        reconciliation adopts a position the bot did not open.
        """
        if self.config.settings.execution.stop_policy != "exchange_stop_limit":
            return
        protected = {
            record.symbol
            for record in self.state.open_orders.values()
            if record.intent == "stop"
        }
        for symbol, position in self.state.positions.items():
            if symbol in protected:
                continue
            try:
                self.execution.place_protective_stop(
                    symbol,
                    position.qty,
                    position.entry_price,
                    self.state,
                    cycle,
                    stop_price=position.stop_price,
                )
                summary["actions"].append(f"placed protective stop {symbol}")
            except ExecutionError as exc:
                log.error("cycle.stop_placement_failed", symbol=symbol, error=str(exc))
                summary["errors"].append(f"stop placement failed {symbol}: {exc}")
                self.alerter.send(
                    Alert(
                        title="Protective stop NOT placed",
                        message=f"{symbol} is open with no exchange-side stop: {exc}",
                        severity=Severity.CRITICAL,
                        fields={"Symbol": symbol},
                    )
                )

    def _scan_entries(self, bars, stats, account, now, cycle, summary) -> None:
        signals = generate_signals(
            bars, self.config.settings.strategy, self.config.settings.universe.benchmark
        )
        atr_by_symbol = self._latest_atr(bars)
        candidates = latest_candidates(signals)
        if candidates.empty:
            log.info("cycle.no_candidates")
            return

        tradable = set(self.config.settings.universe.resolve())
        wanted = [
            symbol
            for symbol in candidates.index.get_level_values("symbol")
            if symbol in tradable and symbol not in self.state.positions
        ]
        if not wanted:
            return

        try:
            quotes = self.market.get_latest_quotes(wanted)
        except DataError as exc:
            log.warning("cycle.no_action_quote_failure", reason=str(exc))
            summary["errors"].append(f"quotes: {exc}")
            return

        for symbol in wanted:
            quote = quotes.get(symbol)
            if quote is None:
                continue
            try:
                self.market.assert_quote_fresh(quote, now)
            except DataError as exc:
                log.warning("cycle.skip_stale_quote", symbol=symbol, reason=str(exc))
                continue

            limit_price = self.execution.entry_limit_price(quote)
            stop_price, target_price = stop_and_target_prices(
                limit_price, self.config.settings.strategy, atr=atr_by_symbol.get(symbol)
            )
            distance = stop_distance_fraction(limit_price, stop_price)
            account_state = self._account_state(account, stats, now)
            sized = (
                account_state.equity
                * self.config.settings.risk.risk_per_trade_pct
                / (distance or self.config.settings.risk.stop_distance_pct)
            ) / limit_price
            decision = self.risk.check(
                ProposedOrder(
                    symbol, "buy", sized, limit_price, intent="entry", stop_price=stop_price
                ),
                account_state,
            )
            if not decision.approved:
                log.info("cycle.entry_rejected", symbol=symbol, reason=decision.reason)
                self.alerter.risk_rejection(symbol, decision.reason)
                summary["rejections"].append(decision.reason)
                continue
            try:
                self.execution.place_entry(decision.order, self.state, cycle)
                self.state.pending_levels[symbol] = [stop_price, target_price]
                self.alerter.trade(
                    symbol, "buy", decision.order.qty, decision.order.limit_price, "entry"
                )
                summary["actions"].append(
                    f"ENTRY {symbol} qty={decision.order.qty:.8g} @ {decision.order.limit_price:.4f}"
                )
            except ExecutionError as exc:
                log.error("cycle.entry_failed", symbol=symbol, error=str(exc))
                summary["errors"].append(f"entry failed {symbol}: {exc}")

    def _latest_atr(self, bars: pd.DataFrame) -> dict[str, float]:
        """Each symbol's ATR on the most recent bar, for stop placement."""
        if self.config.settings.strategy.stop_mode != "atr_multiple" or bars.empty:
            return {}
        series = average_true_range(bars, self.config.settings.strategy.atr_period)
        latest = bars.index.get_level_values("timestamp").max()
        result: dict[str, float] = {}
        for symbol in dict.fromkeys(bars.index.get_level_values("symbol")):
            try:
                value = float(series.loc[(symbol, latest)])
            except KeyError:
                continue
            if value > 0 and value == value:  # finite, not NaN
                result[symbol] = value
        return result

    def _maybe_heartbeat(self, account, now: datetime) -> None:
        hours = self.config.settings.schedule.heartbeat_hours_utc
        if now.hour not in hours or self._last_heartbeat_hour == now.hour:
            return
        self._last_heartbeat_hour = now.hour
        self.alerter.heartbeat(
            account.equity, account.cash, self.state.as_positions(), self.state.realized_pnl_today
        )

    def _log_summary(self, cycle: int, account, summary: dict) -> None:
        log.info(
            "cycle.complete",
            cycle=cycle,
            equity=round(account.equity, 2),
            cash=round(account.cash, 2),
            positions=len(self.state.positions),
            open_orders=len(self.state.open_orders),
            day_pnl=round(self.state.realized_pnl_today, 2),
            day_anchor=round(self.state.day_anchor_equity, 2),
            halted=bool(self.state.halted_until),
            actions=summary["actions"] or None,
            rejections=len(summary["rejections"]) or None,
            errors=summary["errors"] or None,
        )

    # -------------------------------------------------------------- scheduler

    def start(self) -> int:
        from apscheduler.schedulers.blocking import BlockingScheduler

        interval = self.config.settings.schedule.interval_minutes
        try:
            account = self.broker.get_account()
            mode = "LIVE" if self.config.secrets.is_live else "paper"
            self.alerter.startup(account.equity, len(self.state.positions), mode)
        except Exception as exc:  # noqa: BLE001
            log.error("bot.startup_account_failed", error=str(exc))
            self.alerter.exception("startup", exc)
            return 1

        scheduler = BlockingScheduler(timezone="UTC")
        scheduler.add_job(
            self._scheduled_cycle,
            "interval",
            minutes=interval,
            id="run_cycle",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=int(interval * 60 / 2),
        )

        def handle_signal(signum, _frame):
            log.warning("bot.signal", signal=signum)
            scheduler.shutdown(wait=False)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        log.info(
            "bot.started",
            interval_minutes=interval,
            mode="LIVE" if self.config.secrets.is_live else "paper",
            universe=len(self.config.settings.universe.resolve()),
        )
        # Crypto is 24/7, so there is no market-open wait: run immediately.
        self._scheduled_cycle()
        if self._shutdown:
            return 0
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("bot.stopped")
        return 0

    def _scheduled_cycle(self) -> None:
        self.run_cycle()
        if self._shutdown:
            log.warning("bot.exiting", reason="kill switch")
            sys.exit(0)


def cli() -> int:
    parser = argparse.ArgumentParser(description="Unattended crypto trading bot.")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument(
        "--flatten", action="store_true", help="cancel all orders, close all positions, exit"
    )
    parser.add_argument(
        "--offline-demo",
        action="store_true",
        help="run one cycle against a fake broker and generated data (no keys needed)",
    )
    parser.add_argument("--json-logs", action="store_true", help="JSON log output")
    args = parser.parse_args()

    configure_logging(json_logs=args.json_logs)

    if args.offline_demo:
        from tradebot.offline import run_offline_demo

        return run_offline_demo()

    config = get_config()
    try:
        config.secrets.require_alpaca()
    except RuntimeError as exc:
        log.error("bot.credentials_missing", error=str(exc))
        return 2

    if config.secrets.is_live:
        log.warning(
            "bot.LIVE_MODE",
            msg="ALPACA_PAPER=false — this bot will trade REAL MONEY",
        )

    bot = TradingBot(config)
    if args.flatten:
        return bot.flatten_and_exit("--flatten requested")
    if args.once:
        bot.run_cycle()
        return 0
    return bot.start()


if __name__ == "__main__":
    sys.exit(cli())
