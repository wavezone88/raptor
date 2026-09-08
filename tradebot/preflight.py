"""Step-1 connectivity check: prove credentials work before anything trades.

    python -m tradebot.preflight

Prints account equity and a live BTC/USD quote. Read-only — places no orders.
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
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoLatestQuoteRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass

    benchmark = config.settings.universe.benchmark

    try:
        trading = TradingClient(
            api_key=config.secrets.alpaca_api_key,
            secret_key=config.secrets.alpaca_secret_key,
            paper=config.secrets.alpaca_paper,
        )
        account = trading.get_account()
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
        crypto_status=str(getattr(account, "crypto_status", "unknown")),
    )

    # Crypto trades 24/7, so there is no clock gate — but confirm the account
    # is actually enabled for crypto before anything else.
    if str(getattr(account, "crypto_status", "")).upper() not in {"ACTIVE", "ACCOUNTSTATUS.ACTIVE"}:
        log.warning(
            "preflight.crypto_not_active",
            msg="account crypto_status is not ACTIVE — crypto orders will be rejected",
        )

    try:
        assets = trading.get_all_assets(GetAssetsRequest(asset_class=AssetClass.CRYPTO))
        tradable = sorted(a.symbol for a in assets if a.tradable)
    except Exception as exc:  # noqa: BLE001
        log.warning("preflight.asset_discovery_failed", error=str(exc))
        tradable = []

    universe = config.settings.universe.resolve(tradable)
    log.info(
        "preflight.universe",
        discovered=len(tradable),
        after_exclusions=len(universe),
        sample=universe[:8],
        source="broker" if tradable else "settings.yaml snapshot",
    )

    try:
        # Crypto market data on Alpaca is free and needs no subscription tier.
        data = CryptoHistoricalDataClient(
            api_key=config.secrets.alpaca_api_key,
            secret_key=config.secrets.alpaca_secret_key,
        )
        quotes = data.get_crypto_latest_quote(
            CryptoLatestQuoteRequest(symbol_or_symbols=benchmark)
        )
        quote = quotes[benchmark]
    except Exception as exc:  # noqa: BLE001
        log.error("preflight.data_api_failed", error=str(exc), error_type=type(exc).__name__)
        return 1

    log.info(
        "preflight.quote",
        symbol=benchmark,
        bid=str(quote.bid_price),
        ask=str(quote.ask_price),
        bid_size=str(quote.bid_size),
        ask_size=str(quote.ask_size),
        timestamp=str(quote.timestamp),
    )
    log.info("preflight.ok", msg="credentials, trading API and crypto data API all reachable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
