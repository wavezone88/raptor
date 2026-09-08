"""Tests for data.py — caching, freshness, and the liquidity metric.

The spec's rule is "if any data is stale or any API call fails, the cycle takes
no action and logs why". These tests pin the raising behaviour that rule
depends on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tradebot.config import Config, Secrets, Settings
from tradebot.data import DataError, MarketData, Quote, average_dollar_volume, cache_filename

from .conftest import bars_for_symbol, stack

NOW = datetime(2025, 1, 30, 12, 0, tzinfo=timezone.utc)


class FakeBarsResponse:
    def __init__(self, df):
        self.df = df


class FakeDataClient:
    """Records calls so tests can assert the cache avoided the network."""

    def __init__(self, bars=None, quotes=None, fail=False):
        self._bars = bars
        self._quotes = quotes or {}
        self.fail = fail
        self.bar_calls = 0
        self.quote_calls = 0

    def get_crypto_bars(self, request):
        self.bar_calls += 1
        if self.fail:
            raise ConnectionError("alpaca unreachable")
        return FakeBarsResponse(self._bars)

    def get_crypto_latest_quote(self, request):
        self.quote_calls += 1
        if self.fail:
            raise ConnectionError("alpaca unreachable")
        return self._quotes


@pytest.fixture
def config(tmp_path, monkeypatch):
    settings = Settings.load()
    monkeypatch.setattr(type(settings), "cache_path", property(lambda self: tmp_path / "cache"))
    return Config(secrets=Secrets(alpaca_api_key="k", alpaca_secret_key="s"), settings=settings)


# ------------------------------------------------------------------ freshness


def test_stale_quote_raises(config):
    market = MarketData(config, data_client=FakeDataClient())
    stale = Quote("BTC/USD", 100.0, 101.0, NOW - timedelta(minutes=45))
    with pytest.raises(DataError, match="stale quote"):
        market.assert_quote_fresh(stale, NOW)


def test_fresh_quote_passes(config):
    market = MarketData(config, data_client=FakeDataClient())
    market.assert_quote_fresh(Quote("BTC/USD", 100.0, 101.0, NOW - timedelta(minutes=5)), NOW)


def test_quote_with_no_bid_or_ask_is_unusable(config):
    """A one-sided book must not be priced off; it would size the order wrong."""
    market = MarketData(config, data_client=FakeDataClient())
    with pytest.raises(DataError, match="unusable quote"):
        market.assert_quote_fresh(Quote("BTC/USD", 0.0, 101.0, NOW), NOW)


def test_stale_bars_raise_and_name_the_symbol(config):
    """Crypto is 24/7, so a bar gap is an outage rather than a weekend."""
    market = MarketData(config, data_client=FakeDataClient())
    old = stack(BTC_USD=bars_for_symbol([100.0] * 3, start=NOW - timedelta(days=30)))
    with pytest.raises(DataError, match="stale bars for BTC/USD"):
        market.assert_bars_fresh(old, ["BTC/USD"], NOW)


def test_fresh_bars_pass(config):
    market = MarketData(config, data_client=FakeDataClient())
    recent = stack(BTC_USD=bars_for_symbol([100.0] * 3, start=NOW - timedelta(days=2)))
    market.assert_bars_fresh(recent, ["BTC/USD"], NOW)


def test_missing_symbol_raises(config):
    market = MarketData(config, data_client=FakeDataClient())
    bars = stack(BTC_USD=bars_for_symbol([100.0] * 3, start=NOW - timedelta(days=1)))
    with pytest.raises(DataError, match="no bars for ETH/USD"):
        market.assert_bars_fresh(bars, ["BTC/USD", "ETH/USD"], NOW)


def test_empty_bars_raise(config):
    market = MarketData(config, data_client=FakeDataClient())
    with pytest.raises(DataError, match="no bars"):
        market.assert_bars_fresh(pd.DataFrame(), ["BTC/USD"], NOW)


# --------------------------------------------------------------- api failure


def test_bar_fetch_failure_raises_dataerror(config):
    """Any client exception becomes DataError so run_cycle() can no-op."""
    market = MarketData(config, data_client=FakeDataClient(fail=True))
    with pytest.raises(DataError, match="bar fetch failed"):
        market.get_daily_bars(["BTC/USD"], NOW - timedelta(days=10), NOW)


def test_quote_fetch_failure_raises_dataerror(config):
    market = MarketData(config, data_client=FakeDataClient(fail=True))
    with pytest.raises(DataError, match="quote fetch failed"):
        market.get_latest_quotes(["BTC/USD"])


# -------------------------------------------------------------------- cache


def test_bars_are_cached_and_reused(config):
    bars = stack(BTC_USD=bars_for_symbol([100.0] * 10, start=NOW - timedelta(days=9)))
    client = FakeDataClient(bars=bars)
    market = MarketData(config, data_client=client)

    first = market.get_daily_bars(["BTC/USD"], NOW - timedelta(days=9), NOW)
    assert client.bar_calls == 1
    assert len(first) == 10
    assert (market.cache_dir / cache_filename("BTC/USD", "1Day")).exists()

    # Same window again: the cache covers it, so no second network call.
    second = MarketData(config, data_client=client).get_daily_bars(
        ["BTC/USD"], NOW - timedelta(days=9), NOW
    )
    assert client.bar_calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_force_refresh_bypasses_cache(config):
    bars = stack(BTC_USD=bars_for_symbol([100.0] * 10, start=NOW - timedelta(days=9)))
    client = FakeDataClient(bars=bars)
    market = MarketData(config, data_client=client)
    market.get_daily_bars(["BTC/USD"], NOW - timedelta(days=9), NOW)
    market.get_daily_bars(["BTC/USD"], NOW - timedelta(days=9), NOW, force_refresh=True)
    assert client.bar_calls == 2


def test_refetched_bar_overwrites_cached_copy(config):
    """A revised final bar replaces the cached one instead of duplicating it."""
    start = NOW - timedelta(days=9)
    original = stack(BTC_USD=bars_for_symbol([100.0] * 10, start=start))
    market = MarketData(config, data_client=FakeDataClient(bars=original))
    market.get_daily_bars(["BTC/USD"], start, NOW)

    revised = stack(BTC_USD=bars_for_symbol([100.0] * 9 + [999.0], start=start))
    market = MarketData(config, data_client=FakeDataClient(bars=revised))
    result = market.get_daily_bars(["BTC/USD"], start, NOW, force_refresh=True)

    assert len(result) == 10, "no duplicate rows"
    assert result["close"].iloc[-1] == 999.0


def test_cache_filename_is_path_safe():
    assert cache_filename("BTC/USD", "1Day") == "BTC-USD__1Day.parquet"
    assert "/" not in cache_filename("ETH/USD", "1Day")


# --------------------------------------------------------------- liquidity


def test_average_dollar_volume_replaces_share_count():
    """Share counts are meaningless across a $100k coin and a $0.00002 one;
    dollars turned over is comparable."""
    expensive = bars_for_symbol([100_000.0] * 30, volumes=[100.0] * 30)   # $10M/day
    cheap = bars_for_symbol([0.00002] * 30, volumes=[5e11] * 30)          # $10M/day
    bars = stack(BTC_USD=expensive, SHIB_USD=cheap)
    dollar_volume = average_dollar_volume(bars, window=30)
    assert dollar_volume.xs("BTC/USD", level="symbol").iloc[-1] == pytest.approx(1e7)
    assert dollar_volume.xs("SHIB/USD", level="symbol").iloc[-1] == pytest.approx(1e7)


def test_average_dollar_volume_is_trailing():
    bars = stack(BTC_USD=bars_for_symbol([100.0] * 5, volumes=[10.0] * 4 + [1_000.0]))
    result = average_dollar_volume(bars, window=5)
    assert result.iloc[0] == pytest.approx(1_000.0)
    assert result.iloc[-1] == pytest.approx((4 * 1_000.0 + 100_000.0) / 5)
