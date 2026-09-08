"""Step-1 connectivity check: prove credentials work before anything trades.

    python -m tradebot.preflight

Prints account equity and a live SPY quote. Read-only — places no orders.
"""

from __future__ import annotations

import sys

from tradebot.config import get_config
from tradebot.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


def main() -> int:
    config = get_config()
    configure_logging(json_logs=False)

    log.info("preflight.start", **config.secrets.redacted())
    if config.secrets.is_live:
        log.warning("preflight.live_mode", msg="ALPACA_PAPER=false — these are LIVE keys")

    try:
        config.secrets.require_alpaca()
    except RuntimeError as exc:
        log.error("preflight.credentials_missing", error=str(exc))
        return 2

    # Imported here so the module loads (and gives a clean error) without
    # network or credentials present.
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient

    benchmark = config.settings.universe.benchmark

    try:
        trading = TradingClient(
            api_key=config.secrets.alpaca_api_key,
            secret_key=config.secrets.alpaca_secret_key,
            paper=config.secrets.alpaca_paper,
        )
        account = trading.get_account()
        clock = trading.get_clock()
    except Exception as exc:  # noqa: BLE001 — surface any failure to the operator
        log.error("preflight.trading_api_failed", error=str(exc), error_type=type(exc).__name__)
        return 1

    log.info(
        "preflight.account",
        account_number=account.account_number,
        status=str(account.status),
        equity=str(account.equity),
        cash=str(account.cash),
        buying_power=str(account.buying_power),
        pattern_day_trader=account.pattern_day_trader,
        daytrade_count=getattr(account, "daytrade_count", None),
        shorting_enabled=account.shorting_enabled,
    )
    log.info(
        "preflight.clock",
        is_open=clock.is_open,
        next_open=str(clock.next_open),
        next_close=str(clock.next_close),
    )

    try:
        data = StockHistoricalDataClient(
            api_key=config.secrets.alpaca_api_key,
            secret_key=config.secrets.alpaca_secret_key,
        )
        quotes = data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=benchmark))
        quote = quotes[benchmark]
    except Exception as exc:  # noqa: BLE001
        log.error("preflight.data_api_failed", error=str(exc), error_type=type(exc).__name__)
        return 1

    log.info(
        "preflight.quote",
        symbol=benchmark,
        bid=str(quote.bid_price),
        ask=str(quote.ask_price),
        bid_size=quote.bid_size,
        ask_size=quote.ask_size,
        timestamp=str(quote.timestamp),
    )
    log.info("preflight.ok", msg="credentials, trading API and data API all reachable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
