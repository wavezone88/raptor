# Small image for a small VM. Python 3.11 to match the project floor.
FROM python:3.11-slim

# Fail fast and log immediately — buffered stdout hides the last lines when a
# container is killed, which is exactly when you need them.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build tooling for numpy/pandas/numba wheels that lack a slim-compatible build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY tradebot ./tradebot
RUN pip install --upgrade pip && pip install .

COPY settings.yaml ./

# Never run the bot as root. State and cache must be writable by this user.
RUN useradd --create-home --uid 10001 tradebot \
 && mkdir -p /app/state /app/data_cache /app/logs \
 && chown -R tradebot:tradebot /app
USER tradebot

# Secrets come from the environment (docker run --env-file .env), never baked in.
# Persist state and cache with:
#   -v tradebot-state:/app/state -v tradebot-cache:/app/data_cache
VOLUME ["/app/state", "/app/data_cache"]

# A cycle that has not run in ~3 intervals means the scheduler is wedged.
HEALTHCHECK --interval=5m --timeout=30s --start-period=2m --retries=3 \
  CMD python -c "import json,sys,time; \
s=json.load(open('/app/state/bot_state.json')); \
from datetime import datetime,timezone; \
last=s.get('last_cycle_at'); \
sys.exit(0 if last and (datetime.now(timezone.utc)-datetime.fromisoformat(last)).total_seconds() < 2700 else 1)" \
  || exit 1

ENTRYPOINT ["python", "-m", "tradebot.main"]
CMD ["--json-logs"]
