"""Webhook alerts (Discord, with Slack supported by the same interface).

Two rules govern everything here:

  * Alerting must never crash the bot. A failed webhook is logged and
    swallowed. Losing an alert is bad; losing the process that manages open
    positions because a webhook 500'd is worse.
  * Anything that changes money or stops trading alerts immediately: trades,
    risk rejections, halts, reconciliation mismatches, exceptions. Heartbeats
    carry the routine picture.

Identical alerts inside the dedupe window are suppressed so a repeating
per-cycle condition does not bury the one new message that matters.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from tradebot.config import Config, get_config
from tradebot.logging_setup import get_logger

log = get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 10


class Severity(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    CRITICAL = "critical"


# Discord embed colours.
COLOURS = {
    Severity.INFO: 0x3498DB,
    Severity.SUCCESS: 0x2ECC71,
    Severity.WARNING: 0xE67E22,
    Severity.CRITICAL: 0xE74C3C,
}

EMOJI = {
    Severity.INFO: "ℹ️",
    Severity.SUCCESS: "✅",
    Severity.WARNING: "⚠️",
    Severity.CRITICAL: "\U0001f6a8",
}


@dataclass
class Alert:
    title: str
    message: str
    severity: Severity = Severity.INFO
    fields: dict[str, Any] | None = None

    def dedupe_key(self) -> str:
        return f"{self.severity}|{self.title}|{self.message}"


class Alerter:
    """Sends alerts to a Discord or Slack incoming webhook."""

    def __init__(self, config: Config | None = None, session: Any | None = None):
        self.config = config or get_config()
        self.channel = self.config.settings.alerts.channel
        self.dedupe_window = self.config.settings.alerts.dedupe_window_seconds
        self._recent: dict[str, float] = {}
        self._session = session

    @property
    def enabled(self) -> bool:
        return bool(self.config.secrets.alert_webhook_url)

    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    # ------------------------------------------------------------------ send

    def send(self, alert: Alert) -> bool:
        """Deliver one alert. Returns whether it was sent. Never raises."""
        if not self.enabled:
            log.debug("alert.disabled", title=alert.title)
            return False

        now = time.monotonic()
        key = alert.dedupe_key()
        last = self._recent.get(key)
        if last is not None and (now - last) < self.dedupe_window:
            log.debug("alert.deduped", title=alert.title)
            return False

        payload = (
            self._discord_payload(alert)
            if self.channel == "discord"
            else self._slack_payload(alert)
        )
        try:
            response = self.session.post(
                self.config.secrets.alert_webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code >= 300:
                log.warning(
                    "alert.rejected",
                    status=response.status_code,
                    body=str(response.text)[:200],
                    title=alert.title,
                )
                return False
        except Exception as exc:  # noqa: BLE001 — an alert must never crash the bot
            log.warning(
                "alert.failed", error=str(exc), error_type=type(exc).__name__, title=alert.title
            )
            return False

        self._recent[key] = now
        log.info("alert.sent", title=alert.title, severity=alert.severity.value)
        return True

    # --------------------------------------------------------------- payloads

    def _discord_payload(self, alert: Alert) -> dict:
        embed: dict[str, Any] = {
            "title": f"{EMOJI[alert.severity]} {alert.title}",
            "description": alert.message[:4000],
            "color": COLOURS[alert.severity],
        }
        if alert.fields:
            embed["fields"] = [
                {"name": str(name), "value": f"`{value}`", "inline": True}
                for name, value in list(alert.fields.items())[:25]
            ]
        mode = "LIVE" if self.config.secrets.is_live else "paper"
        embed["footer"] = {"text": f"tradebot · {mode}"}
        return {"embeds": [embed]}

    def _slack_payload(self, alert: Alert) -> dict:
        lines = [f"{EMOJI[alert.severity]} *{alert.title}*", alert.message]
        if alert.fields:
            lines += [f"• *{name}*: `{value}`" for name, value in alert.fields.items()]
        mode = "LIVE" if self.config.secrets.is_live else "paper"
        lines.append(f"_tradebot · {mode}_")
        return {"text": "\n".join(lines)}

    # ------------------------------------------------------- semantic alerts

    def heartbeat(
        self, equity: float, cash: float, positions: Iterable[Any], day_pnl: float
    ) -> bool:
        positions = list(positions)
        summary = (
            ", ".join(f"{p.symbol} {p.qty:.8g} @ {p.entry_price:,.4f}" for p in positions)
            or "flat"
        )
        return self.send(
            Alert(
                title="Heartbeat",
                message=summary,
                severity=Severity.INFO,
                fields={
                    "Equity": f"${equity:,.2f}",
                    "Cash": f"${cash:,.2f}",
                    "Day P&L": f"${day_pnl:,.2f}",
                    "Positions": len(positions),
                },
            )
        )

    def trade(
        self, symbol: str, side: str, qty: float, price: float, intent: str, pnl: float | None = None
    ) -> bool:
        fields = {
            "Symbol": symbol,
            "Side": side.upper(),
            "Qty": f"{qty:.8g}",
            "Price": f"${price:,.4f}",
            "Notional": f"${qty * price:,.2f}",
            "Reason": intent,
        }
        if pnl is not None:
            fields["Realized P&L"] = f"${pnl:,.2f}"
        severity = Severity.SUCCESS if (pnl is None or pnl >= 0) else Severity.WARNING
        return self.send(
            Alert(
                title=f"Trade · {side.upper()} {symbol}",
                message=f"{intent} filled",
                severity=severity,
                fields=fields,
            )
        )

    def risk_rejection(self, symbol: str, reason: str) -> bool:
        return self.send(
            Alert(
                title="Risk rejection",
                message=reason,
                severity=Severity.WARNING,
                fields={"Symbol": symbol},
            )
        )

    def halt(self, reason: str, until: Any, flattened: int = 0) -> bool:
        return self.send(
            Alert(
                title="TRADING HALTED",
                message=reason,
                severity=Severity.CRITICAL,
                fields={"Halted until": str(until), "Positions flattened": flattened},
            )
        )

    def reconciliation_mismatch(self, details: str, fields: dict | None = None) -> bool:
        return self.send(
            Alert(
                title="Reconciliation mismatch",
                message=details,
                severity=Severity.CRITICAL,
                fields=fields,
            )
        )

    def exception(self, where: str, error: Exception) -> bool:
        return self.send(
            Alert(
                title="Unhandled exception",
                message=f"{type(error).__name__}: {error}"[:1500],
                severity=Severity.CRITICAL,
                fields={"Where": where},
            )
        )

    def kill_switch(self, flattened: int, cancelled: int) -> bool:
        return self.send(
            Alert(
                title="KILL SWITCH — shutting down",
                message="STOP file found or --flatten requested. Flattening and exiting.",
                severity=Severity.CRITICAL,
                fields={"Positions flattened": flattened, "Orders cancelled": cancelled},
            )
        )

    def startup(self, equity: float, positions: int, mode: str) -> bool:
        return self.send(
            Alert(
                title="Bot started",
                message=f"Running in {mode} mode.",
                severity=Severity.INFO,
                fields={"Equity": f"${equity:,.2f}", "Open positions": positions},
            )
        )
