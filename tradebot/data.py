"""Market data: Alpaca crypto bars and quotes, cached to parquet.

Two independent freshness checks, because this strategy trades on DAILY bars
while run_cycle() fires every 15 minutes:

  * quotes must be fresher than interval_minutes * stale_quote_interval_multiple
    (they price the order we are about to send)
  * daily bars must be no older than max_bar_age_calendar_days
    (they drive the signal; crypto is 24/7 so a gap is an outage, not a weekend)

Applying the quote rule to daily bars would stall the bot permanently — a daily
bar is almost always more than 30 minutes old. Every failure here raises
DataError so the caller can no-op the cycle and log why, per the spec's
"if any data is stale or any API call fails, take no action" rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from tradebot.config import Config, get_config
from tradebot.logging_setup import get_logger

log = get_logger(__name__)

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """Raised for any data problem that must stop the cycle: API failure,
    missing symbols, or stale bars/quotes."""


@dataclass(frozen=True)
class Quote:
    """Broker-agnostic quote. Only what execution.py actually needs."""

    symbol: str
    bid: float
    ask: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.ask or self.bid

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        return (self.ask - self.bid) / mid if mid > 0 else float("inf")


def cache_filename(symbol: str, timeframe: str) -> str:
    """BTC/USD -> BTC-USD__1Day.parquet (slashes are not path-safe)."""
    return f"{symbol.replace('/', '-').replace(':', '-')}__{timeframe}.parquet"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | pd.Timestamp) -> pd.Timestamp:
    """Timestamp in UTC, whether the input is naive or already aware.

    pd.Timestamp(x, tz=...) raises on an aware input, so localize and convert
    explicitly rather than guessing which case we are in.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _normalise_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce an Alpaca bars frame to the (symbol, timestamp) shape we use."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    frame = frame.copy()
    missing = [c for c in BAR_COLUMNS if c not in frame.columns]
    if missing:
        raise DataError(f"bars missing columns {missing}")
    frame = frame[BAR_COLUMNS]
    frame.index = frame.index.set_names(["symbol", "timestamp"])
    return frame.sort_index()


class MarketData:
    """Fetches and caches bars and quotes.

    Clients are injected rather than constructed inline so tests can pass
    fakes without touching the network.
    """

    def __init__(
        self,
        config: Config | None = None,
        data_client: Any | None = None,
        trading_client: Any | None = None,
    ) -> None:
        self.config = config or get_config()
        self._data_client = data_client
        self._trading_client = trading_client
        self.cache_dir = Path(self.config.settings.cache_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ clients

    @property
    def data_client(self) -> Any:
        if self._data_client is None:
            from alpaca.data.historical import CryptoHistoricalDataClient

            self.config.secrets.require_alpaca()
            self._data_client = CryptoHistoricalDataClient(
                api_key=self.config.secrets.alpaca_api_key,
                secret_key=self.config.secrets.alpaca_secret_key,
            )
        return self._data_client

    @property
    def trading_client(self) -> Any:
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient

            self.config.secrets.require_alpaca()
            self._trading_client = TradingClient(
                api_key=self.config.secrets.alpaca_api_key,
                secret_key=self.config.secrets.alpaca_secret_key,
                paper=self.config.secrets.alpaca_paper,
            )
        return self._trading_client

    # ---------------------------------------------------------------- discovery

    def discover_universe(self) -> list[str]:
        """Alpaca's live tradable crypto list, exclusions applied.

        Never raises: a discovery failure falls back to the settings.yaml
        snapshot rather than emptying the universe.
        """
        universe = self.config.settings.universe
        if not universe.discover_from_broker:
            return universe.resolve()

        try:
            from alpaca.trading.enums import AssetClass
            from alpaca.trading.requests import GetAssetsRequest

            assets = self.trading_client.get_all_assets(
                GetAssetsRequest(asset_class=AssetClass.CRYPTO)
            )
            discovered = [a.symbol for a in assets if getattr(a, "tradable", False)]
        except Exception as exc:  # noqa: BLE001 — fall back, never crash the cycle
            log.warning(
                "data.discovery_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                fallback="settings.yaml snapshot",
            )
            discovered = []

        resolved = universe.resolve(discovered)
        log.info(
            "data.universe_resolved",
            count=len(resolved),
            source="broker" if discovered else "settings.yaml snapshot",
        )
        return resolved

    # --------------------------------------------------------------------- bars

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / cache_filename(symbol, self.config.settings.data.bar_timeframe)

    def _read_cache(self, symbol: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 — a corrupt cache must not be fatal
            log.warning("data.cache_read_failed", symbol=symbol, error=str(exc))
            return None
        if frame.empty:
            return None
        return frame.sort_index()

    def _write_cache(self, symbol: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        try:
            frame.sort_index().to_parquet(self._cache_path(symbol))
        except Exception as exc:  # noqa: BLE001
            log.warning("data.cache_write_failed", symbol=symbol, error=str(exc))

    def _fetch_bars(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        try:
            response = self.data_client.get_crypto_bars(
                CryptoBarsRequest(
                    symbol_or_symbols=list(symbols),
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise DataError(f"bar fetch failed: {type(exc).__name__}: {exc}") from exc
        return _normalise_bars(getattr(response, "df", None))

    def get_daily_bars(
        self,
        symbols: Iterable[str],
        start: datetime,
        end: datetime | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Daily bars for `symbols` over [start, end], cache-backed.

        Only the uncached tail is fetched. Cached and fresh rows are combined
        with the fetched rows taking precedence, so a revised bar overwrites a
        stale cached copy rather than duplicating it.
        """
        symbols = [s.upper() for s in symbols]
        if not symbols:
            return pd.DataFrame(columns=BAR_COLUMNS)
        end = end or _utc_now()

        cached: dict[str, pd.DataFrame] = {}
        needs_fetch: list[str] = []
        earliest_needed = end

        for symbol in symbols:
            frame = None if force_refresh else self._read_cache(symbol)
            if frame is None or frame.empty:
                needs_fetch.append(symbol)
                earliest_needed = start
                continue
            cached[symbol] = frame
            last = pd.Timestamp(frame.index.max()).to_pydatetime()
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            covers_start = _as_utc(frame.index.min()) <= _as_utc(start)
            if not covers_start:
                needs_fetch.append(symbol)
                earliest_needed = start
            elif last < end - timedelta(days=1):
                needs_fetch.append(symbol)
                # Re-fetch from one day before the cached end so a partially
                # formed final bar is replaced, not appended to.
                earliest_needed = min(earliest_needed, last - timedelta(days=1))

        fetched = pd.DataFrame(columns=BAR_COLUMNS)
        if needs_fetch:
            log.info(
                "data.fetching_bars",
                symbols=len(needs_fetch),
                start=earliest_needed.isoformat(),
                end=end.isoformat(),
            )
            fetched = self._fetch_bars(needs_fetch, earliest_needed, end)

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            parts = []
            if symbol in cached:
                parts.append(cached[symbol])
            if not fetched.empty and symbol in set(
                fetched.index.get_level_values("symbol")
            ):
                parts.append(fetched.xs(symbol, level="symbol"))
            if not parts:
                continue
            # Later frames win on duplicate timestamps.
            merged = pd.concat(parts)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self._write_cache(symbol, merged)
            merged = merged.copy()
            merged["symbol"] = symbol
            frames.append(merged.set_index("symbol", append=True).reorder_levels(
                ["symbol", "timestamp"]
            ))

        if not frames:
            raise DataError(f"no bars returned for any of {symbols[:5]}")

        combined = pd.concat(frames).sort_index()
        window = combined.index.get_level_values("timestamp") >= _as_utc(start)
        return combined[window]

    # ------------------------------------------------------------------- quotes

    def get_latest_quotes(self, symbols: Iterable[str]) -> dict[str, Quote]:
        from alpaca.data.requests import CryptoLatestQuoteRequest

        symbols = [s.upper() for s in symbols]
        if not symbols:
            return {}
        try:
            raw = self.data_client.get_crypto_latest_quote(
                CryptoLatestQuoteRequest(symbol_or_symbols=symbols)
            )
        except Exception as exc:  # noqa: BLE001
            raise DataError(f"quote fetch failed: {type(exc).__name__}: {exc}") from exc

        quotes: dict[str, Quote] = {}
        for symbol, quote in raw.items():
            timestamp = quote.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            quotes[symbol] = Quote(
                symbol=symbol,
                bid=float(quote.bid_price or 0.0),
                ask=float(quote.ask_price or 0.0),
                timestamp=timestamp,
            )
        return quotes

    # --------------------------------------------------------------- freshness

    @property
    def max_quote_age(self) -> timedelta:
        settings = self.config.settings
        return timedelta(
            minutes=settings.schedule.interval_minutes
            * settings.data.stale_quote_interval_multiple
        )

    @property
    def max_bar_age(self) -> timedelta:
        return timedelta(days=self.config.settings.data.max_bar_age_calendar_days)

    def assert_quote_fresh(self, quote: Quote, now: datetime | None = None) -> None:
        now = now or _utc_now()
        age = now - quote.timestamp
        if age > self.max_quote_age:
            raise DataError(
                f"stale quote for {quote.symbol}: {age.total_seconds():.0f}s old, "
                f"limit {self.max_quote_age.total_seconds():.0f}s"
            )
        if quote.ask <= 0 or quote.bid <= 0:
            raise DataError(f"unusable quote for {quote.symbol}: bid={quote.bid} ask={quote.ask}")

    def assert_bars_fresh(
        self, bars: pd.DataFrame, symbols: Iterable[str] | None = None, now: datetime | None = None
    ) -> None:
        """Every requested symbol must have a recent bar. Raises on the first
        that does not, naming it, so the cycle log says exactly what stalled."""
        now = now or _utc_now()
        if bars.empty:
            raise DataError("no bars available")
        limit = _as_utc(now - self.max_bar_age)
        symbols = list(symbols) if symbols is not None else list(
            dict.fromkeys(bars.index.get_level_values("symbol"))
        )
        available = set(bars.index.get_level_values("symbol"))
        for symbol in symbols:
            if symbol not in available:
                raise DataError(f"no bars for {symbol}")
            last = _as_utc(bars.xs(symbol, level="symbol").index.max())
            if last < limit:
                raise DataError(
                    f"stale bars for {symbol}: last bar {last.isoformat()}, "
                    f"limit {limit.isoformat()}"
                )


def average_dollar_volume(bars: pd.DataFrame, window: int = 30) -> pd.Series:
    """Trailing mean of close * volume, per symbol.

    This is the crypto replacement for the equity spec's "30-day average volume
    above 5M shares": share counts are meaningless when one unit is $100k and
    another is $0.00002, but dollars turned over per day is comparable.
    """
    if bars.empty:
        return pd.Series(dtype="float64")
    dollar_volume = bars["close"] * bars["volume"]
    return dollar_volume.groupby(level="symbol").transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
