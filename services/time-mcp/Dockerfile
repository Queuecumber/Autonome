FROM python:3.12-slim

WORKDIR /app

COPY services/time-mcp/ ./services/time-mcp/

RUN pip install --no-cache-dir ./services/time-mcp

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "time_mcp.mcp_server"]
