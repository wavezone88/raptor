"""Configuration: secrets from .env, tunables from settings.yaml.

Two separate sources on purpose. Anything in settings.yaml is safe to commit
and review in a diff; anything in .env is a credential and is gitignored.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "settings.yaml"

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
LIVE_TRADING_URL = "https://api.alpaca.markets"


class Secrets(BaseSettings):
    """Credentials, loaded from .env or the process environment."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    alert_webhook_url: str = ""

    @property
    def trading_base_url(self) -> str:
        return PAPER_TRADING_URL if self.alpaca_paper else LIVE_TRADING_URL

    @property
    def is_live(self) -> bool:
        return not self.alpaca_paper

    def require_alpaca(self) -> None:
        """Raise if credentials are missing, with an actionable message."""
        missing = [
            name
            for name, value in (
                ("ALPACA_API_KEY", self.alpaca_api_key),
                ("ALPACA_SECRET_KEY", self.alpaca_secret_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing {', '.join(missing)}. Copy .env.example to .env and fill "
                "in your Alpaca keys."
            )

    def redacted(self) -> dict[str, str]:
        """Safe-to-log view. Never log the raw model."""

        def mask(value: str) -> str:
            return f"{value[:4]}...{value[-2:]}" if len(value) > 8 else ("set" if value else "unset")

        return {
            "alpaca_api_key": mask(self.alpaca_api_key),
            "alpaca_secret_key": "set" if self.alpaca_secret_key else "unset",
            "alpaca_paper": str(self.alpaca_paper),
            "alert_webhook_url": "set" if self.alert_webhook_url else "unset",
            "mode": "LIVE" if self.is_live else "paper",
        }


class UniverseSettings(BaseModel):
    etfs: list[str]
    equities: list[str]
    benchmark: str

    @property
    def symbols(self) -> list[str]:
        """Tradable universe, de-duplicated, order preserved."""
        seen: dict[str, None] = {}
        for symbol in [*self.etfs, *self.equities]:
            seen.setdefault(symbol.upper(), None)
        return list(seen)

    @property
    def symbols_with_benchmark(self) -> list[str]:
        """Everything we need bars for, including the ranking benchmark."""
        symbols = self.symbols
        benchmark = self.benchmark.upper()
        return symbols if benchmark in symbols else [*symbols, benchmark]


class StrategySettings(BaseModel):
    trend_sma_days: int = Field(gt=0)
    pullback_lookback_days: int = Field(gt=0)
    pullback_min_pct: float = Field(ge=0)
    pullback_max_pct: float = Field(gt=0)
    volume_avg_days: int = Field(gt=0)
    volume_multiple: float = Field(gt=0)
    rel_strength_lookback_days: int = Field(gt=0)
    stop_pct: float = Field(gt=0)
    target_pct: float = Field(gt=0)
    time_stop_days: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_band(self) -> "StrategySettings":
        if self.pullback_min_pct >= self.pullback_max_pct:
            raise ValueError("pullback_min_pct must be below pullback_max_pct")
        return self


class RiskSettings(BaseModel):
    starting_capital: float = Field(gt=0)
    max_open_positions: int = Field(gt=0)
    max_position_pct_equity: float = Field(gt=0, le=1)
    risk_per_trade_pct: float = Field(gt=0, le=1)
    stop_distance_pct: float = Field(gt=0, lt=1)
    daily_loss_limit_pct: float = Field(gt=0, le=1)
    weekly_loss_limit_pct: float = Field(gt=0, le=1)
    max_day_trades_in_5_business_days: int = Field(ge=0)
    min_avg_volume_30d: float = Field(ge=0)
    min_price: float = Field(ge=0)
    fractional_qty_decimals: int = Field(ge=0, le=9)


class ExecutionSettings(BaseModel):
    entry_limit_buffer_pct: float = Field(ge=0)
    cancel_unfilled_after_cycles: int = Field(gt=0)
    fractional_policy: Literal["reject", "allow_bot_managed_stop"]


class ScheduleSettings(BaseModel):
    interval_minutes: int = Field(gt=0)
    pre_close_cancel_minutes: int = Field(gt=0)
    timezone: str


class DataSettings(BaseModel):
    cache_dir: str
    bar_timeframe: str
    stale_bar_interval_multiple: float = Field(gt=0)


class AlertSettings(BaseModel):
    channel: Literal["discord", "slack"]
    dedupe_window_seconds: int = Field(ge=0)


class BacktestSettings(BaseModel):
    years: int = Field(gt=0)
    cost_per_side_pct: float = Field(ge=0)
    equity_curve_png: str


class StateSettings(BaseModel):
    dir: str
    file: str


class Settings(BaseModel):
    """Everything from settings.yaml, validated."""

    universe: UniverseSettings
    strategy: StrategySettings
    risk: RiskSettings
    execution: ExecutionSettings
    schedule: ScheduleSettings
    data: DataSettings
    alerts: AlertSettings
    backtest: BacktestSettings
    state: StateSettings

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Settings":
        settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
        if not settings_path.exists():
            raise FileNotFoundError(f"settings file not found: {settings_path}")
        with settings_path.open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    @property
    def state_path(self) -> Path:
        return PROJECT_ROOT / self.state.dir / self.state.file

    @property
    def cache_path(self) -> Path:
        return PROJECT_ROOT / self.data.cache_dir

    @property
    def stop_file_path(self) -> Path:
        """Kill switch. If this file exists, the bot flattens and exits."""
        return PROJECT_ROOT / "STOP"


class Config(BaseModel):
    """Secrets plus settings, the single object passed around the bot."""

    model_config = {"arbitrary_types_allowed": True}

    secrets: Secrets
    settings: Settings

    @classmethod
    def load(cls, settings_path: Path | str | None = None) -> "Config":
        return cls(secrets=Secrets(), settings=Settings.load(settings_path))


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide config. Cached so .env is read once."""
    return Config.load()
