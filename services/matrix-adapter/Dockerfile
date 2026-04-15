FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libolm-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/matrix-adapter/ ./services/matrix-adapter/

RUN pip install --no-cache-dir ./services/matrix-adapter

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "matrix_adapter.mcp_server"]
