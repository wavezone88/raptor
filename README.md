# tradebot

An unattended, rules-only trading bot for Alpaca crypto. Deterministic: no LLM
calls anywhere in the trading loop, no discretion, no manual overrides. Every
order passes a risk manager that can veto or resize it.

**Read [Before you run this](#before-you-run-this) first.** There are three
things about this build you need to know before it touches money.

---

## Contents

- [Before you run this](#before-you-run-this)
- [What it does](#what-it-does)
- [Setup](#setup)
- [Running it](#running-it)
- [Switching to live money](#switching-to-live-money)
- [Stopping it](#stopping-it)
- [Alerts: what each one means](#alerts-what-each-one-means)
- [Risk rules](#risk-rules)
- [What protects you if the process dies](#what-protects-you-if-the-process-dies)
- [Deployment](#deployment)
- [Project layout](#project-layout)
- [Testing](#testing)

---

## Before you run this

### 1. The backtest has not been run on real data

The strategy has **never been validated against real market history**. The
session that built this had no network access to Alpaca or any market-data
provider, so only a synthetic-data run was possible — and synthetic data proves
the plumbing works, nothing more.

Run this before you trade anything, paper or live:

```bash
python -m tradebot.backtest
```

One result from the synthetic run does carry over, because it is arithmetic
rather than a property of the data: at **453 trades over two years and ~0.25%
per side**, every round trip must clear roughly **0.5%** before it earns
anything. Net expectancy came out at 0.010% per trade — the edge was almost
exactly consumed by fees. Expect the real backtest to be tight for the same
structural reason.

`backtest.py` prints its own verdict, states plainly if the strategy loses money
after costs, and names parameters to revisit. **Change one at a time.** Searching
combinations until the curve looks good fits the parameters to that specific
history and will not survive live.

### 2. The strategy parameters were designed for equities, not crypto

This started as a US equities spec and was pivoted to crypto. The entry and exit
parameters were deliberately **left unchanged** so you can see what they actually
do rather than what a tuned version pretends to:

| Parameter | Value | Concern on crypto |
|---|---|---|
| `stop_pct` | 4% | Crypto daily volatility is several times an equity index's. A 4% stop is close to noise — the synthetic run exited 53.6% of trades at the stop. |
| Pullback band | 2–4% | Roughly a single day's move for a liquid coin, so it fires on noise rather than on a genuine pullback. |
| `rel_strength` benchmark | BTC/USD | Most of the universe is heavily BTC-correlated, so "max 3 positions" is closer to one bet in triplicate than to three independent ones. |

These are honest defaults, not recommended ones. Revisit them using the backtest.

### 3. There are no bracket orders on crypto

The original spec called for bracket orders so the stop rests at the exchange
and survives the bot process dying. **Alpaca supports bracket and OCO orders for
equities only.** Crypto accepts market, limit and stop-limit.

The closest available equivalent is implemented: a **resting stop-limit sell**
placed immediately after each entry fills. That does survive process death. The
**take-profit cannot rest at the exchange** and is enforced only by `run_cycle()`.
See [What protects you if the process dies](#what-protects-you-if-the-process-dies).

---

## What it does

**Universe** — Alpaca's tradable USD crypto pairs, discovered at startup with a
snapshot in `settings.yaml` as fallback. Stablecoins excluded: a 2–4% pullback in
USDT/USD is a depeg, not a setup.

**Strategy** — mean reversion with a trend filter, long only.

| | Rule |
|---|---|
| Trend filter | Close above the 20-day SMA |
| Entry | Price 2–4% below its 5-day high, trend holds, volume above the prior 10-day average |
| Ranking | Highest 3-month return relative to BTC |
| Exit | −4% stop, +8% target, or a 5-day time stop |
| Never | Add to a losing position |

**Cycle** — every 15 minutes, 24/7:

```
kill switch → reconcile → halts → exits → stops → stale orders
            → scan entries → risk check → execute → persist → summary
```

Exits run before entries so risk is reduced before it is added.

---

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/wavezone88/raptor.git
cd raptor

uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"
# or: python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

cp .env.example .env
```

Fill in `.env`:

```ini
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/yyy
```

Paper keys come from the [Alpaca paper dashboard](https://app.alpaca.markets/paper/dashboard/overview).
The Discord webhook comes from *Server Settings → Integrations → Webhooks → New Webhook*.

`.env` is gitignored. Never commit it.

Verify credentials without placing an order:

```bash
python -m tradebot.preflight
```

This prints account equity, cash, crypto status, the discovered universe, and a
live BTC/USD quote. If it fails, everything else will too.

---

## Running it

```bash
python -m tradebot.backtest              # ALWAYS do this first
python -m tradebot.backtest --synthetic  # pipeline check, no keys needed

python -m tradebot.main --offline-demo   # a full cycle, fake broker, no keys
python -m tradebot.main --once           # one real cycle against paper, then exit
python -m tradebot.main                  # run on the schedule, forever
python -m tradebot.main --json-logs      # JSON output for a log collector
```

Suggested order: backtest → read the verdict → `--offline-demo` → `--once` on
paper → leave it running on paper for a couple of weeks → only then consider live.

`--offline-demo` runs the entire production code path against an in-memory
broker and generated bars. Useful for watching a cycle before trusting it.

Everything is configured in `settings.yaml`, which is commented rule by rule.

---

## Switching to live money

**Two changes. Nothing in the code.**

1. Put your live keys in `.env` (from the
   [live dashboard](https://app.alpaca.markets/live/dashboard/overview) — they are
   different keys, not the paper ones)
2. Set `ALPACA_PAPER=false`

```ini
ALPACA_API_KEY=your_LIVE_key
ALPACA_SECRET_KEY=your_LIVE_secret
ALPACA_PAPER=false
```

The bot logs `bot.LIVE_MODE` at startup and every alert is footered `LIVE`
instead of `paper`, so you can tell at a glance which one is running.

Before you flip it:

- [ ] The backtest has been run on **real** bars and you accept the result
- [ ] It has run on paper long enough to have taken and exited real trades
- [ ] `state/bot_state.json` has been deleted or archived — paper state must not
      carry into a live account
- [ ] You have confirmed which stop policy is active and accept it
- [ ] Your alert webhook works and reaches a device you actually check
- [ ] You are prepared to lose the whole balance

At the configured `starting_capital: 50.0`, the risk rules size each position at
**$12.50** notional. That is a live-fire test, not an investment. To scale up,
change `risk.starting_capital` in `settings.yaml` — but note that live equity is
read from Alpaca, so this setting only affects backtests and reference sizing.

---

## Stopping it

**Kill switch** — the bot checks for a `STOP` file at the top of every cycle. If
found, it cancels all orders, closes all positions, alerts, and exits.

```bash
touch STOP
```

It takes effect on the next cycle (up to 15 minutes). To act now:

```bash
python -m tradebot.main --flatten
```

Cancels everything, closes everything, exits immediately.

**Stop without closing positions** — `systemctl stop tradebot`, or Ctrl-C.
Positions stay open, and any resting stop-limit orders stay at the exchange.
**Remember `STOP` blocks startup**: delete the file before restarting.

---

## Alerts: what each one means

All alerts go to your Discord webhook. Identical alerts are suppressed within
`alerts.dedupe_window_seconds` (default 300) so a repeating condition cannot bury
a new one.

| Alert | Severity | What happened | What to do |
|---|---|---|---|
| **Bot started** | info | Process came up; shows equity and open positions | Confirm the mode footer says what you expect (`paper` / `LIVE`) |
| **Heartbeat** | info | Routine status at the configured UTC hours: equity, cash, day P&L, positions | Nothing. Its *absence* is the signal — no heartbeat means the process is down |
| **Trade · BUY** | success | An entry limit was submitted | Nothing |
| **Trade · SELL** | success / warning | A position was closed; shows realized P&L and the reason (`stop`, `target`, `time_stop`, `flatten`). Warning when the P&L is negative | Nothing |
| **Risk rejection** | warning | The risk manager vetoed a proposed order; the reason is verbatim | Usually normal (position cap, liquidity floor, minimum notional). Investigate only if you expected the trade |
| **TRADING HALTED** | critical | Daily −5% or weekly −10% breached. Everything was flattened; no new entries until the timestamp shown | Look at what happened before re-enabling. The halt clears itself |
| **Reconciliation mismatch** | critical | Local state disagreed with Alpaca. Alpaca won and state was corrected | **Investigate.** Means an order filled the bot didn't know about, a position vanished, or something else touched the account |
| **Protective stop NOT placed** | critical | A position is open with **no exchange-side stop** | **Act now.** Place a stop manually in Alpaca, or flatten |
| **Unhandled exception** | critical | Something threw inside `run_cycle()`. The bot logged it and continued | Read the logs. Repeated instances mean it is not trading |
| **KILL SWITCH** | critical | `STOP` file found or `--flatten` used. Flattened and exiting | Expected if you did it. If you didn't, find out who created `STOP` |

**The alert you should worry about most is the one that stops arriving.** If
heartbeats go quiet, the process is dead and only the resting stop-limits are
protecting you.

---

## Risk rules

Every rule is in `settings.yaml`, enforced in `risk.py`, and covered by a test
that fails if the rule is deleted (verified by deleting each in turn, not just by
passing).

| Rule | Default |
|---|---|
| Max open positions | 3 |
| Max single position | 40% of equity |
| Position size | (equity × 1%) ÷ 4% stop, floored to 8 decimals |
| Daily loss limit | −5% from the UTC-day anchor → flatten + halt to next UTC midnight |
| Weekly loss limit | −10% from the Monday anchor → flatten + halt to next Monday |
| Round trips per day | 3 (churn and fee brake) |
| Liquidity floor | $10M 30-day average dollar volume |
| Cash | Crypto is non-marginable — cash is a hard ceiling |
| Stale data | Any stale bar or quote, or any API failure → the cycle takes no action and logs why |

Two invariants worth knowing:

- **Sells are never blocked.** Every rule can stop the bot opening or adding
  risk; none can stop it reducing risk. A rule that could veto an exit could trap
  you in a losing position.
- **Unknown liquidity is a rejection**, never an assumption of good liquidity.

### What crypto changed from the equities original

Removed, because they do not apply: the pattern-day-trader guard (FINRA
round-trip rules don't cover crypto), unsettled-cash handling (crypto settles
instantly), and market-hours gating (24/7, so no clock endpoint and no pre-close
job).

Added or adapted: `max_round_trips_per_day` replaces PDT as a churn brake; the
share-count liquidity filter became dollar volume, since share counts are
meaningless across a $100k coin and a $0.00002 one; loss anchors moved to UTC
boundaries because a 24/7 market has no session open; and the assumed cost rose
from 0.05% to 0.25% per side.

---

## What protects you if the process dies

Ranked by what actually survives:

| Protection | Survives process death? |
|---|---|
| Resting stop-limit sell at −4% | **Yes** — it lives at Alpaca |
| Take-profit at +8% | **No** — bot-managed only |
| Time stop (5 days) | **No** |
| Daily / weekly loss halts | **No** |
| Kill switch | **No** — requires a running process |

With `stop_policy: exchange_stop_limit` (the default), a dead bot still has a
stop at the exchange on every position. With `stop_policy: bot_managed`,
**nothing protects an open position** if the process stops. That policy exists so
the choice is explicit rather than accidental; do not use it unattended.

This is why the systemd unit uses `Restart=always` and why a missing heartbeat
matters.

---

## Deployment

### systemd

```bash
sudo useradd --system --create-home --home-dir /opt/tradebot tradebot
sudo -u tradebot git clone https://github.com/wavezone88/raptor.git /opt/tradebot
cd /opt/tradebot
sudo -u tradebot python3.11 -m venv .venv
sudo -u tradebot .venv/bin/pip install .

sudo -u tradebot cp .env.example .env
sudo -u tradebot nano .env
sudo chmod 600 .env                       # secrets: owner-readable only

sudo cp tradebot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tradebot
journalctl -u tradebot -f
```

`Restart=always` with `RestartSec=15` — the bot holds positions, so getting it
back up matters more than backing off politely.

### Docker

```bash
docker build -t tradebot .
docker run -d --name tradebot --restart unless-stopped \
  --env-file .env \
  -v tradebot-state:/app/state \
  -v tradebot-cache:/app/data_cache \
  tradebot
docker logs -f tradebot
```

Mount the state volume. Without it, every restart loses the day/week loss
anchors — and a bot that forgets it was down 4% today gets a fresh 5% of rope.

The healthcheck fails if no cycle has completed in ~45 minutes (three intervals).

---

## Project layout

```
tradebot/
  config.py        pydantic-settings: secrets from .env, tunables from settings.yaml
  logging_setup.py structlog, JSON to stdout
  data.py          Alpaca bars/quotes, parquet cache, freshness checks
  strategy.py      pure functions — DataFrame in, signals out
  risk.py          RiskManager.check(): approve, resize, or reject with a reason
  execution.py     idempotent orders, reconciliation, protective stops
  state.py         persisted state, atomic writes
  alerts.py        Discord/Slack webhooks
  main.py          scheduler and run_cycle()
  backtest.py      vectorbt statistics over a risk-aware simulation
  offline.py       fake broker for --offline-demo
  preflight.py     credential and connectivity check
tests/
  test_strategy.py test_risk.py test_execution.py test_data.py test_backtest.py
settings.yaml      every tunable, commented rule by rule
```

`strategy.py` is asset-agnostic OHLCV math with no network, clock, or broker
dependency — which is what lets the backtester and the live loop run *identical*
logic instead of two implementations that drift apart.

---

## Testing

```bash
pytest                 # full suite
pytest tests/test_risk.py -v
```

All fixtures are synthetic — no network, no recorded market data — so every rule
can be tested at its exact boundary.

The risk rules are verified by **mutation**: each of the seven core rules was
deleted in turn and confirmed to break the suite. A test that passes only because
the code happens to be right today is not much of a test.

---

## Known limitations

- **Not validated on real data.** See [Before you run this](#before-you-run-this).
- **No take-profit at the exchange.** Alpaca has no crypto OCO.
- **Universe is BTC-correlated**, so the 3-position cap diversifies less than it
  looks like it does.
- **`vectorbt` does statistics, not order accounting.** `Portfolio.from_orders`
  takes one order per (bar, symbol) and cannot represent an entry and a stop-out
  on the same bar; netting them erases the round trip. The bar-by-bar simulation
  in `backtest.py` is authoritative.
- **`plotly` is pinned `<6`** — vectorbt 1.1.0 references `scattermapbox`, removed
  in plotly 6.
- **Backtest fills are optimistic on liquidity.** Entries fill at the next bar's
  open and stops at the stop price; real slippage on a thin pair will be worse
  than the flat 0.25% assumption.

---

## License

Private project. No warranty. **Trading carries risk of total loss — you are
responsible for anything this software does with your money.**
