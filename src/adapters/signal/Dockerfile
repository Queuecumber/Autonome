FROM python:3.12-slim

WORKDIR /app

COPY src/adapters/signal/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "adapters.signal.mcp_server"]
