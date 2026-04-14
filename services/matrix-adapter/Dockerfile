FROM python:3.12-slim

WORKDIR /app

COPY services/matrix-adapter/ ./services/matrix-adapter/

RUN pip install --no-cache-dir ./services/matrix-adapter

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "matrix_adapter.mcp_server"]
