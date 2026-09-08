"""Tests for execution.py.

The two properties that matter most here:

  * idempotency — a retry after an ambiguous timeout must not open a second
    position
  * reconciliation — Alpaca is the source of truth and every divergence is
    reported, because a bot trading against a stale idea of its own positions
    is more dangerous than one that is not trading
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradebot.config import Config, Secrets, Settings
from tradebot.data import Quote
from tradebot.execution import (
    AccountSnapshot,
    AlpacaBroker,
    BrokerOrder,
    BrokerPosition,
    ExecutionEngine,
    ExecutionError,
    make_client_order_id,
)
from tradebot.risk import ProposedOrder
from tradebot.state import BotState, OrderRecord

NOW = datetime(2025, 6, 11, 12, 0, tzinfo=timezone.utc)


class FakeBroker:
    """Records every call so tests can assert on what was sent."""

    def __init__(self, positions=None, orders=None, fail_on=None):
        self.positions = list(positions or [])
        self.orders = list(orders or [])
        self.submitted: list[dict] = []
        self.cancelled: list[str] = []
        self.fail_on = fail_on or set()
        self._next = 0

    def _id(self) -> str:
        self._next += 1
        return f"broker-{self._next}"

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=50.0, cash=50.0, buying_power=50.0)

    def get_positions(self):
        return list(self.positions)

    def get_open_orders(self):
        return [o for o in self.orders if o.is_open]

    def _make(self, kind, symbol, side, qty, **extra) -> BrokerOrder:
        if kind in self.fail_on:
            raise ExecutionError(f"{kind} rejected by test")
        self.submitted.append(
            {"kind": kind, "symbol": symbol, "side": side, "qty": qty, **extra}
        )
        order = BrokerOrder(
            id=self._id(),
            client_order_id=extra.get("client_order_id", ""),
            symbol=symbol,
            side=side,
            qty=qty,
            status="new",
            order_type=kind,
            limit_price=extra.get("limit_price"),
        )
        self.orders.append(order)
        return order

    def submit_limit(self, symbol, side, qty, limit_price, client_order_id):
        return self._make(
            "limit", symbol, side, qty, limit_price=limit_price, client_order_id=client_order_id
        )

    def submit_stop_limit(self, symbol, side, qty, stop_price, limit_price, client_order_id):
        return self._make(
            "stop_limit", symbol, side, qty,
            stop_price=stop_price, limit_price=limit_price, client_order_id=client_order_id,
        )

    def submit_market(self, symbol, side, qty, client_order_id):
        return self._make("market", symbol, side, qty, client_order_id=client_order_id)

    def cancel_order(self, order_id: str) -> None:
        if "cancel" in self.fail_on:
            raise ExecutionError("cancel rejected by test")
        self.cancelled.append(order_id)
        self.orders = [o for o in self.orders if o.id != order_id]


@pytest.fixture
def config():
    return Config(secrets=Secrets(), settings=Settings.load())


@pytest.fixture
def engine(config):
    return ExecutionEngine(FakeBroker(), config)


def order_record(symbol="BTC/USD", intent="entry", cycle=1, broker_id="broker-1"):
    return OrderRecord(
        client_order_id=make_client_order_id(symbol, "buy", intent, cycle, NOW),
        symbol=symbol,
        side="buy",
        qty=0.1,
        limit_price=100.0,
        intent=intent,
        submitted_cycle=cycle,
        submitted_at=NOW.isoformat(),
        broker_order_id=broker_id,
    )


# ------------------------------------------------------------- idempotency


def test_same_logical_order_produces_the_same_id():
    """The whole retry-safety story rests on this."""
    first = make_client_order_id("BTC/USD", "buy", "entry", 7, NOW)
    second = make_client_order_id("BTC/USD", "buy", "entry", 7, NOW)
    assert first == second


@pytest.mark.parametrize(
    ("field", "value"),
    [("symbol", "ETH/USD"), ("side", "sell"), ("intent", "stop"), ("cycle", 8)],
)
def test_id_changes_when_any_component_changes(field, value):
    base = {"symbol": "BTC/USD", "side": "buy", "intent": "entry", "cycle": 7}
    assert make_client_order_id(**base, moment=NOW) != make_client_order_id(
        **{**base, field: value}, moment=NOW
    )


def test_id_is_slash_free_and_within_broker_length_limits():
    order_id = make_client_order_id("BTC/USD", "buy", "entry", 1, NOW)
    assert "/" not in order_id
    assert len(order_id) <= 128


def test_resubmitting_an_entry_in_the_same_cycle_does_not_place_a_second_order(engine):
    """A retry after an ambiguous timeout must not double the position."""
    state = BotState()
    order = ProposedOrder("BTC/USD", "buy", 0.1, 100.0, intent="entry")
    engine.place_entry(order, state, cycle=1)
    engine.place_entry(order, state, cycle=1)
    assert len(engine.broker.submitted) == 1


def test_a_later_cycle_may_place_a_new_entry(engine):
    state = BotState()
    order = ProposedOrder("BTC/USD", "buy", 0.1, 100.0, intent="entry")
    engine.place_entry(order, state, cycle=1)
    engine.place_entry(order, state, cycle=2)
    assert len(engine.broker.submitted) == 2


def test_duplicate_id_at_the_broker_is_not_retried(config):
    """A duplicate means our previous attempt landed — that is the guarantee
    working, not a transient error to retry through."""

    class DuplicateClient:
        def submit_order(self, request):
            raise RuntimeError("client_order_id already exists")

    broker = AlpacaBroker(config, client=DuplicateClient())
    with pytest.raises(ExecutionError, match="already exists at the broker"):
        broker.submit_limit("BTC/USD", "buy", 0.1, 100.0, "tb-x")


# ----------------------------------------------------------- reconciliation


def test_reconcile_adopts_a_position_the_bot_did_not_know_about(config):
    broker = FakeBroker(positions=[BrokerPosition("BTC/USD", 0.5, 100.0)])
    engine = ExecutionEngine(broker, config)
    state = BotState()

    report = engine.reconcile(state)

    assert report.has_mismatch
    assert report.adopted_positions == ["BTC/USD"]
    assert state.positions["BTC/USD"].qty == 0.5
    assert state.positions["BTC/USD"].entry_price == 100.0


def test_reconcile_drops_a_position_the_broker_does_not_have(config):
    engine = ExecutionEngine(FakeBroker(), config)
    state = BotState()
    state.record_entry("BTC/USD", 0.5, 100.0, NOW)

    report = engine.reconcile(state)

    assert report.dropped_positions == ["BTC/USD"]
    assert "BTC/USD" not in state.positions


def test_reconcile_lets_the_broker_win_on_a_quantity_mismatch(config):
    broker = FakeBroker(positions=[BrokerPosition("BTC/USD", 0.25, 105.0)])
    engine = ExecutionEngine(broker, config)
    state = BotState()
    state.record_entry("BTC/USD", 0.5, 100.0, NOW)

    report = engine.reconcile(state)

    assert report.qty_mismatches
    assert state.positions["BTC/USD"].qty == 0.25, "Alpaca is the source of truth"
    assert state.positions["BTC/USD"].entry_price == 105.0


def test_reconcile_adopts_an_unknown_broker_order(config):
    broker = FakeBroker(
        orders=[BrokerOrder("b1", "someone-elses-order", "BTC/USD", "buy", 0.1, "new")]
    )
    engine = ExecutionEngine(broker, config)
    state = BotState()

    report = engine.reconcile(state)

    assert report.unknown_orders
    assert "someone-elses-order" in state.open_orders


def test_reconcile_drops_an_order_that_vanished_from_the_broker(config):
    engine = ExecutionEngine(FakeBroker(), config)
    state = BotState()
    record = order_record()
    state.add_order(record)

    report = engine.reconcile(state)

    assert report.vanished_orders == [record.client_order_id]
    assert record.client_order_id not in state.open_orders


def test_reconcile_reports_no_mismatch_when_already_in_sync(config):
    record = order_record()
    broker = FakeBroker(
        positions=[BrokerPosition("BTC/USD", 0.1, 100.0)],
        orders=[BrokerOrder("broker-1", record.client_order_id, "BTC/USD", "buy", 0.1, "new")],
    )
    engine = ExecutionEngine(broker, config)
    state = BotState()
    state.record_entry("BTC/USD", 0.1, 100.0, NOW)
    state.add_order(record)

    report = engine.reconcile(state)

    assert not report.has_mismatch
    assert report.describe() == "in sync"


# ------------------------------------------------------------------ entries


def test_entry_limit_crosses_the_spread_by_the_configured_buffer(engine, config):
    quote = Quote("BTC/USD", bid=99.0, ask=100.0, timestamp=NOW)
    expected = 100.0 * (1 + config.settings.execution.entry_limit_buffer_pct)
    assert engine.entry_limit_price(quote) == pytest.approx(expected)


def test_entry_is_submitted_as_a_limit_order(engine):
    state = BotState()
    engine.place_entry(ProposedOrder("BTC/USD", "buy", 0.1, 100.0, "entry"), state, cycle=1)
    sent = engine.broker.submitted[0]
    assert sent["kind"] == "limit"
    assert sent["side"] == "buy"
    assert sent["limit_price"] == 100.0


# -------------------------------------------------------------------- stops


def test_protective_stop_rests_at_the_configured_stop_distance(engine, config):
    state = BotState()
    state.record_entry("BTC/USD", 0.1, 100.0, NOW)
    engine.place_protective_stop("BTC/USD", 0.1, 100.0, state, cycle=1)

    sent = engine.broker.submitted[0]
    assert sent["kind"] == "stop_limit"
    assert sent["side"] == "sell"
    assert sent["stop_price"] == pytest.approx(96.0)
    # Limit sits below the trigger so a fast move still fills.
    assert sent["limit_price"] < sent["stop_price"]


def test_protective_stop_is_skipped_under_bot_managed_policy(config):
    """The policy exists so the loss of exchange-side protection is explicit
    rather than accidental."""
    config.settings.execution.stop_policy = "bot_managed"
    engine = ExecutionEngine(FakeBroker(), config)
    state = BotState()
    assert engine.place_protective_stop("BTC/USD", 0.1, 100.0, state, cycle=1) is None
    assert engine.broker.submitted == []


def test_stop_is_recorded_against_the_position(engine):
    state = BotState()
    state.record_entry("BTC/USD", 0.1, 100.0, NOW)
    engine.place_protective_stop("BTC/USD", 0.1, 100.0, state, cycle=1)
    assert state.positions["BTC/USD"].stop_order_id is not None


# -------------------------------------------------------------------- exits


def test_exit_is_a_market_order(engine):
    state = BotState()
    state.record_entry("BTC/USD", 0.1, 100.0, NOW)
    engine.place_exit("BTC/USD", 0.1, "stop", state, cycle=1)
    assert engine.broker.submitted[-1]["kind"] == "market"


def test_exit_cancels_the_resting_stop_first(engine):
    """Otherwise the stop and the exit race to sell the same units."""
    state = BotState()
    state.record_entry("BTC/USD", 0.1, 100.0, NOW)
    engine.place_protective_stop("BTC/USD", 0.1, 100.0, state, cycle=1)
    stop_broker_id = engine.broker.submitted[0]
    engine.place_exit("BTC/USD", 0.1, "target", state, cycle=2)

    assert engine.broker.cancelled, "resting stop was not cancelled before the exit"
    assert engine.broker.submitted[-1]["kind"] == "market"
    assert not [r for r in state.open_orders.values() if r.intent == "stop"]


# ------------------------------------------------------------- stale orders


def test_unfilled_entry_is_cancelled_after_the_configured_cycles(engine, config):
    state = BotState()
    record = order_record(cycle=1)
    record.broker_order_id = "broker-1"
    state.add_order(record)
    engine.broker.orders.append(
        BrokerOrder("broker-1", record.client_order_id, "BTC/USD", "buy", 0.1, "new")
    )

    max_cycles = config.settings.execution.cancel_unfilled_after_cycles
    assert engine.cancel_stale_entries(state, cycle=1 + max_cycles - 1) == []
    assert engine.cancel_stale_entries(state, cycle=1 + max_cycles) == ["BTC/USD"]
    assert record.client_order_id not in state.open_orders


def test_a_resting_stop_is_never_cancelled_for_being_old(engine):
    """An old protective stop is doing its job, not going stale."""
    state = BotState()
    stop = order_record(intent="stop", cycle=1)
    state.add_order(stop)
    assert engine.cancel_stale_entries(state, cycle=99) == []
    assert stop.client_order_id in state.open_orders


# ----------------------------------------------------------------- flatten


def test_flatten_cancels_orders_then_sells_everything(config):
    broker = FakeBroker(
        positions=[BrokerPosition("BTC/USD", 0.1, 100.0), BrokerPosition("ETH/USD", 1.0, 50.0)],
        orders=[BrokerOrder("broker-9", "some-order", "BTC/USD", "sell", 0.1, "new")],
    )
    engine = ExecutionEngine(broker, config)
    state = BotState()
    state.record_entry("BTC/USD", 0.1, 100.0, NOW)

    flattened = engine.flatten_all(state, cycle=1)

    assert sorted(flattened) == ["BTC/USD", "ETH/USD"]
    assert "broker-9" in broker.cancelled
    assert all(s["kind"] == "market" and s["side"] == "sell" for s in broker.submitted)
    assert state.positions == {}


def test_flatten_continues_after_one_symbol_fails(config):
    """One unsellable position must not strand the others."""
    broker = FakeBroker(
        positions=[BrokerPosition("BTC/USD", 0.1, 100.0)], fail_on={"market"}
    )
    engine = ExecutionEngine(broker, config)
    assert engine.flatten_all(BotState(), cycle=1) == []


# ------------------------------------------------------------ symbol format


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("BTCUSD", "BTC/USD"), ("ETH/USD", "ETH/USD"), ("SHIBUSD", "SHIB/USD")],
)
def test_broker_symbols_are_normalised(raw, expected):
    """Alpaca reports crypto positions as BTCUSD but takes orders as BTC/USD."""
    assert AlpacaBroker._normalise(raw) == expected


# ------------------------------------------------- levels fixed at entry


def test_protective_stop_uses_the_level_fixed_at_entry(engine):
    """Not a freshly computed one — a stop that drifts is not a stop."""
    state = BotState()
    state.record_entry("BTC/USD", 0.1, 100.0, NOW, stop_price=88.0, target_price=124.0)
    engine.place_protective_stop(
        "BTC/USD", 0.1, 100.0, state, cycle=1, stop_price=88.0
    )
    assert engine.broker.submitted[0]["stop_price"] == pytest.approx(88.0)


def test_protective_stop_computes_a_level_for_an_adopted_position(engine):
    """Reconciliation can adopt a position that has no stored levels."""
    state = BotState()
    engine.place_protective_stop("BTC/USD", 0.1, 100.0, state, cycle=1)
    assert engine.broker.submitted[0]["stop_price"] == pytest.approx(96.0)


def test_reconcile_applies_pending_levels_to_a_newly_filled_position(config):
    """The gap between submitting an entry and seeing the fill must not lose
    the stop that was computed when the order was priced."""
    broker = FakeBroker(positions=[BrokerPosition("BTC/USD", 0.1, 100.0)])
    engine = ExecutionEngine(broker, config)
    state = BotState()
    state.pending_levels["BTC/USD"] = [88.0, 124.0]

    engine.reconcile(state)

    assert state.positions["BTC/USD"].stop_price == 88.0
    assert state.positions["BTC/USD"].target_price == 124.0
    assert "BTC/USD" not in state.pending_levels, "levels should be consumed once applied"


def test_position_levels_survive_a_state_round_trip(tmp_path):
    """A restart must not forget where the stop was."""
    path = tmp_path / "state.json"
    state = BotState()
    state.record_entry("BTC/USD", 0.1, 100.0, NOW, stop_price=88.0, target_price=124.0)
    state.pending_levels["ETH/USD"] = [45.0, 60.0]
    state.save(path)

    restored = BotState.load(path)
    assert restored.positions["BTC/USD"].stop_price == 88.0
    assert restored.positions["BTC/USD"].target_price == 124.0
    assert restored.pending_levels["ETH/USD"] == [45.0, 60.0]
    assert restored.as_positions()[0].stop_price == 88.0
