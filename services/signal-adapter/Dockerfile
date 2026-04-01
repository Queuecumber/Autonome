FROM python:3.12-slim

WORKDIR /app

COPY services/signal-adapter/ ./services/signal-adapter/

RUN pip install --no-cache-dir ./services/signal-adapter

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "adapters.signal.mcp_server"]
