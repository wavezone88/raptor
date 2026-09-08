"""Order placement, reconciliation and the protective stop.

Idempotency: every order carries a client_order_id derived from
(date, symbol, side, intent, cycle). Alpaca rejects a duplicate id, so a retry
after a timeout — where we never learned whether the first attempt landed —
cannot open a second position. This is the single most important property in
this file: without it a network blip during a cycle silently doubles risk.

Reconciliation: Alpaca is the source of truth, always. Local state is a cache.
On startup and at the top of every cycle we compare the two, adopt the broker's
view, and alert loudly on any difference. A bot trading against a stale idea of
its own positions is more dangerous than one that is not trading.

The stop, and what crypto costs us: the spec called for bracket orders so the
stop rests at the exchange and survives the process dying. Alpaca supports
bracket/OCO for equities only — crypto takes market, limit and stop-limit. The
closest available equivalent is a resting stop-limit sell placed immediately
after the entry fills (stop_policy: exchange_stop_limit), which does survive
process death. The take-profit cannot rest at the exchange and is always
enforced by run_cycle(). See README "What protects you if the process dies".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol

from tradebot.config import Config, get_config
from tradebot.data import Quote
from tradebot.logging_setup import get_logger
from tradebot.risk import ProposedOrder
from tradebot.state import BotState, OrderRecord
from tradebot.strategy import stop_and_target_prices

log = get_logger(__name__)


class ExecutionError(RuntimeError):
    """An order operation failed. The cycle logs it and takes no further action."""


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float = 0.0


@dataclass(frozen=True)
class BrokerOrder:
    id: str
    client_order_id: str
    symbol: str
    side: str
    qty: float
    status: str
    order_type: str = "limit"
    limit_price: float | None = None
    filled_qty: float = 0.0
    filled_avg_price: float | None = None

    @property
    def is_open(self) -> bool:
        return self.status.lower() in {"new", "accepted", "partially_filled", "pending_new", "held"}

    @property
    def is_filled(self) -> bool:
        return self.status.lower() == "filled"


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float


class Broker(Protocol):
    """The broker surface the bot needs.

    Kept narrow and explicit so execution logic is testable against a fake, and
    so a different venue could be supported without touching strategy or risk.
    """

    def get_account(self) -> AccountSnapshot: ...
    def get_positions(self) -> list[BrokerPosition]: ...
    def get_open_orders(self) -> list[BrokerOrder]: ...
    def submit_limit(
        self, symbol: str, side: str, qty: float, limit_price: float, client_order_id: str
    ) -> BrokerOrder: ...
    def submit_stop_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        limit_price: float,
        client_order_id: str,
    ) -> BrokerOrder: ...
    def submit_market(
        self, symbol: str, side: str, qty: float, client_order_id: str
    ) -> BrokerOrder: ...
    def cancel_order(self, order_id: str) -> None: ...


@dataclass
class ReconciliationReport:
    """Differences between local state and the broker. Empty means agreement."""

    adopted_positions: list[str] = field(default_factory=list)
    dropped_positions: list[str] = field(default_factory=list)
    qty_mismatches: list[str] = field(default_factory=list)
    unknown_orders: list[str] = field(default_factory=list)
    vanished_orders: list[str] = field(default_factory=list)

    @property
    def has_mismatch(self) -> bool:
        return bool(
            self.adopted_positions
            or self.dropped_positions
            or self.qty_mismatches
            or self.unknown_orders
            or self.vanished_orders
        )

    def describe(self) -> str:
        parts = []
        if self.adopted_positions:
            parts.append(f"positions at broker but not local: {', '.join(self.adopted_positions)}")
        if self.dropped_positions:
            parts.append(f"positions local but not at broker: {', '.join(self.dropped_positions)}")
        if self.qty_mismatches:
            parts.append(f"quantity mismatches: {'; '.join(self.qty_mismatches)}")
        if self.unknown_orders:
            parts.append(f"orders at broker but not local: {', '.join(self.unknown_orders)}")
        if self.vanished_orders:
            parts.append(f"orders local but not at broker: {', '.join(self.vanished_orders)}")
        return " | ".join(parts) or "in sync"


def make_client_order_id(
    symbol: str, side: str, intent: str, cycle: int, moment: datetime | None = None
) -> str:
    """Deterministic id from (date, symbol, side, intent, cycle).

    Deterministic is the whole point: submitting the same logical order twice
    in one cycle produces the same id, and the broker rejects the duplicate
    instead of opening a second position.
    """
    moment = moment or datetime.now(timezone.utc)
    clean = symbol.replace("/", "").replace(":", "").upper()
    return f"tb-{moment.strftime('%Y%m%d')}-{clean}-{side.lower()}-{intent}-{cycle}"


class AlpacaBroker:
    """Broker implementation for Alpaca crypto."""

    def __init__(self, config: Config | None = None, client: Any | None = None):
        self.config = config or get_config()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self.config.secrets.require_alpaca()
            self._client = TradingClient(
                api_key=self.config.secrets.alpaca_api_key,
                secret_key=self.config.secrets.alpaca_secret_key,
                paper=self.config.secrets.alpaca_paper,
            )
        return self._client

    def get_account(self) -> AccountSnapshot:
        try:
            account = self.client.get_account()
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(f"get_account failed: {type(exc).__name__}: {exc}") from exc
        return AccountSnapshot(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
        )

    def get_positions(self) -> list[BrokerPosition]:
        try:
            positions = self.client.get_all_positions()
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(f"get_positions failed: {type(exc).__name__}: {exc}") from exc
        return [
            BrokerPosition(
                symbol=self._normalise(p.symbol),
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(getattr(p, "market_value", 0.0) or 0.0),
            )
            for p in positions
        ]

    def get_open_orders(self) -> list[BrokerOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        try:
            orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(f"get_orders failed: {type(exc).__name__}: {exc}") from exc
        return [self._to_order(o) for o in orders]

    @staticmethod
    def _normalise(symbol: str) -> str:
        """Alpaca returns crypto positions as BTCUSD but takes orders as BTC/USD."""
        symbol = symbol.upper()
        if "/" in symbol:
            return symbol
        for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                return f"{symbol[: -len(quote)]}/{quote}"
        return symbol

    def _to_order(self, order: Any) -> BrokerOrder:
        return BrokerOrder(
            id=str(order.id),
            client_order_id=str(order.client_order_id or ""),
            symbol=self._normalise(str(order.symbol)),
            side=str(order.side).lower().replace("orderside.", ""),
            qty=float(order.qty or 0.0),
            status=str(order.status).lower().replace("orderstatus.", ""),
            order_type=str(getattr(order, "order_type", "limit")).lower(),
            limit_price=float(order.limit_price) if getattr(order, "limit_price", None) else None,
            filled_qty=float(getattr(order, "filled_qty", 0.0) or 0.0),
            filled_avg_price=(
                float(order.filled_avg_price)
                if getattr(order, "filled_avg_price", None)
                else None
            ),
        )

    def submit_limit(self, symbol, side, qty, limit_price, client_order_id) -> BrokerOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        request = LimitOrderRequest(
            symbol=symbol,
            qty=round(qty, self.config.settings.risk.fractional_qty_decimals),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            # Crypto does not accept DAY; GTC is the correct time in force.
            time_in_force=TimeInForce.GTC,
            limit_price=round(limit_price, 2),
            client_order_id=client_order_id,
        )
        return self._submit(request, client_order_id)

    def submit_stop_limit(
        self, symbol, side, qty, stop_price, limit_price, client_order_id
    ) -> BrokerOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopLimitOrderRequest

        request = StopLimitOrderRequest(
            symbol=symbol,
            qty=round(qty, self.config.settings.risk.fractional_qty_decimals),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
            limit_price=round(limit_price, 2),
            client_order_id=client_order_id,
        )
        return self._submit(request, client_order_id)

    def submit_market(self, symbol, side, qty, client_order_id) -> BrokerOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, self.config.settings.risk.fractional_qty_decimals),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )
        return self._submit(request, client_order_id)

    def _submit(self, request: Any, client_order_id: str) -> BrokerOrder:
        try:
            return self._to_order(self.client.submit_order(request))
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            # A duplicate id means our previous attempt did land. That is the
            # idempotency guarantee working, not an error to retry through.
            if "client_order_id" in message and (
                "exist" in message.lower() or "duplicate" in message.lower()
            ):
                raise ExecutionError(
                    f"duplicate client_order_id {client_order_id}: the order already exists "
                    "at the broker; not resubmitting"
                ) from exc
            raise ExecutionError(f"submit failed: {type(exc).__name__}: {exc}") from exc

    def cancel_order(self, order_id: str) -> None:
        try:
            self.client.cancel_order_by_id(order_id)
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(f"cancel failed for {order_id}: {exc}") from exc


class ExecutionEngine:
    """Places orders, reconciles state, and maintains the protective stop."""

    def __init__(self, broker: Broker, config: Config | None = None):
        self.broker = broker
        self.config = config or get_config()

    # ---------------------------------------------------------- reconcile

    def reconcile(self, state: BotState) -> ReconciliationReport:
        """Make local state match the broker. Alpaca always wins.

        Returns a report of every difference found so the caller can alert.
        """
        report = ReconciliationReport()
        broker_positions = {p.symbol: p for p in self.broker.get_positions()}
        broker_orders = self.broker.get_open_orders()
        broker_by_client_id = {o.client_order_id: o for o in broker_orders if o.client_order_id}

        # Positions the broker has that we do not know about.
        for symbol, position in broker_positions.items():
            local = state.positions.get(symbol)
            if local is None:
                report.adopted_positions.append(symbol)
                state.record_entry(
                    symbol,
                    position.qty,
                    position.avg_entry_price,
                    datetime.now(timezone.utc),
                )
            elif abs(local.qty - position.qty) > 1e-9:
                report.qty_mismatches.append(
                    f"{symbol} local={local.qty:.8g} broker={position.qty:.8g}"
                )
                local.qty = position.qty
                local.entry_price = position.avg_entry_price

        # Positions we think we have that the broker does not.
        for symbol in list(state.positions):
            if symbol not in broker_positions:
                report.dropped_positions.append(symbol)
                state.positions.pop(symbol, None)

        # Orders.
        for client_order_id in list(state.open_orders):
            if client_order_id not in broker_by_client_id:
                report.vanished_orders.append(client_order_id)
                state.drop_order(client_order_id)
        for client_order_id, order in broker_by_client_id.items():
            if client_order_id not in state.open_orders:
                report.unknown_orders.append(f"{order.symbol}:{client_order_id}")
                state.add_order(
                    OrderRecord(
                        client_order_id=client_order_id,
                        symbol=order.symbol,
                        side=order.side,
                        qty=order.qty,
                        limit_price=order.limit_price or 0.0,
                        intent="adopted",
                        submitted_cycle=state.cycle,
                        submitted_at=datetime.now(timezone.utc).isoformat(),
                        broker_order_id=order.id,
                        status=order.status,
                    )
                )

        log.info(
            "execution.reconciled",
            broker_positions=len(broker_positions),
            broker_open_orders=len(broker_orders),
            mismatch=report.has_mismatch,
            detail=report.describe(),
        )
        return report

    # ------------------------------------------------------------- entries

    def entry_limit_price(self, quote: Quote) -> float:
        """Cross the spread by a small buffer so a limit entry actually fills."""
        buffer = self.config.settings.execution.entry_limit_buffer_pct
        return quote.ask * (1.0 + buffer)

    def place_entry(
        self, order: ProposedOrder, state: BotState, cycle: int
    ) -> OrderRecord | None:
        client_order_id = make_client_order_id(order.symbol, "buy", "entry", cycle)
        if client_order_id in state.open_orders:
            log.info("execution.entry_already_submitted", client_order_id=client_order_id)
            return state.open_orders[client_order_id]

        broker_order = self.broker.submit_limit(
            symbol=order.symbol,
            side="buy",
            qty=order.qty,
            limit_price=order.limit_price,
            client_order_id=client_order_id,
        )
        record = OrderRecord(
            client_order_id=client_order_id,
            symbol=order.symbol,
            side="buy",
            qty=order.qty,
            limit_price=order.limit_price,
            intent="entry",
            submitted_cycle=cycle,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            broker_order_id=broker_order.id,
            status=broker_order.status,
        )
        state.add_order(record)
        log.info(
            "execution.entry_submitted",
            symbol=order.symbol,
            qty=order.qty,
            limit=order.limit_price,
            notional=round(order.notional, 2),
            client_order_id=client_order_id,
        )
        return record

    # --------------------------------------------------------------- stops

    def place_protective_stop(
        self, symbol: str, qty: float, entry_price: float, state: BotState, cycle: int
    ) -> OrderRecord | None:
        """Rest a stop-limit sell at the exchange, if policy allows.

        This is what survives the process dying. Under bot_managed policy no
        exchange-side protection exists at all and the position is only as safe
        as the process.
        """
        if self.config.settings.execution.stop_policy != "exchange_stop_limit":
            log.warning(
                "execution.no_exchange_stop",
                symbol=symbol,
                policy=self.config.settings.execution.stop_policy,
                msg="stop is bot-managed only; it does not survive process death",
            )
            return None

        stop_price, _ = stop_and_target_prices(entry_price, self.config.settings.strategy)
        slippage = self.config.settings.execution.stop_limit_slippage_pct
        limit_price = stop_price * (1.0 - slippage)
        client_order_id = make_client_order_id(symbol, "sell", "stop", cycle)
        if client_order_id in state.open_orders:
            return state.open_orders[client_order_id]

        broker_order = self.broker.submit_stop_limit(
            symbol=symbol,
            side="sell",
            qty=qty,
            stop_price=stop_price,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        record = OrderRecord(
            client_order_id=client_order_id,
            symbol=symbol,
            side="sell",
            qty=qty,
            limit_price=limit_price,
            intent="stop",
            submitted_cycle=cycle,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            broker_order_id=broker_order.id,
            status=broker_order.status,
        )
        state.add_order(record)
        if symbol in state.positions:
            state.positions[symbol].stop_order_id = broker_order.id
        log.info(
            "execution.stop_placed",
            symbol=symbol,
            stop=round(stop_price, 4),
            limit=round(limit_price, 4),
            qty=qty,
        )
        return record

    # --------------------------------------------------------------- exits

    def place_exit(
        self, symbol: str, qty: float, reason: str, state: BotState, cycle: int
    ) -> OrderRecord | None:
        """Close a position at market.

        Exits go market, not limit: an unfilled exit is an unmanaged position,
        and paying the spread is cheaper than not getting out.
        """
        self.cancel_stop_for(symbol, state)
        client_order_id = make_client_order_id(symbol, "sell", reason, cycle)
        if client_order_id in state.open_orders:
            return state.open_orders[client_order_id]

        broker_order = self.broker.submit_market(
            symbol=symbol, side="sell", qty=qty, client_order_id=client_order_id
        )
        record = OrderRecord(
            client_order_id=client_order_id,
            symbol=symbol,
            side="sell",
            qty=qty,
            limit_price=0.0,
            intent=reason,
            submitted_cycle=cycle,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            broker_order_id=broker_order.id,
            status=broker_order.status,
        )
        state.add_order(record)
        log.info("execution.exit_submitted", symbol=symbol, qty=qty, reason=reason)
        return record

    def cancel_stop_for(self, symbol: str, state: BotState) -> None:
        """Cancel a resting stop before selling, or the exit double-sells."""
        for client_order_id, record in list(state.open_orders.items()):
            if record.symbol == symbol and record.intent == "stop" and record.broker_order_id:
                try:
                    self.broker.cancel_order(record.broker_order_id)
                    log.info("execution.stop_cancelled", symbol=symbol)
                except ExecutionError as exc:
                    # Already gone (it filled, or was cancelled) is fine.
                    log.warning("execution.stop_cancel_failed", symbol=symbol, error=str(exc))
                state.drop_order(client_order_id)

    # ---------------------------------------------------------- maintenance

    def cancel_stale_entries(self, state: BotState, cycle: int) -> list[str]:
        """Cancel entry limits unfilled beyond the configured cycle budget.

        A stale entry is a stale price: the pullback we were buying has moved.
        """
        max_cycles = self.config.settings.execution.cancel_unfilled_after_cycles
        cancelled: list[str] = []
        for record in state.stale_orders(cycle, max_cycles):
            if not record.broker_order_id:
                state.drop_order(record.client_order_id)
                continue
            try:
                self.broker.cancel_order(record.broker_order_id)
                cancelled.append(record.symbol)
                log.info(
                    "execution.stale_entry_cancelled",
                    symbol=record.symbol,
                    cycles_open=record.cycles_open(cycle),
                )
            except ExecutionError as exc:
                log.warning(
                    "execution.cancel_failed", symbol=record.symbol, error=str(exc)
                )
            state.drop_order(record.client_order_id)
        return cancelled

    def cancel_all_orders(self, state: BotState) -> int:
        cancelled = 0
        for order in self.broker.get_open_orders():
            try:
                self.broker.cancel_order(order.id)
                cancelled += 1
            except ExecutionError as exc:
                log.warning("execution.cancel_failed", order_id=order.id, error=str(exc))
        state.open_orders.clear()
        return cancelled

    def flatten_all(self, state: BotState, cycle: int) -> list[str]:
        """Cancel everything, then market-sell every position.

        Order matters: a resting stop must be cancelled before the market sell,
        or the two compete to sell the same units.
        """
        self.cancel_all_orders(state)
        flattened: list[str] = []
        for position in self.broker.get_positions():
            if position.qty <= 0:
                continue
            client_order_id = make_client_order_id(
                position.symbol, "sell", "flatten", cycle
            )
            try:
                self.broker.submit_market(
                    symbol=position.symbol,
                    side="sell",
                    qty=position.qty,
                    client_order_id=client_order_id,
                )
                flattened.append(position.symbol)
                log.info("execution.flattened", symbol=position.symbol, qty=position.qty)
            except ExecutionError as exc:
                log.error("execution.flatten_failed", symbol=position.symbol, error=str(exc))
        state.positions.clear()
        return flattened
