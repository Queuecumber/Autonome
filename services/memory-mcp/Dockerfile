FROM python:3.12-slim

WORKDIR /app

COPY services/memory-mcp/ ./services/memory-mcp/

RUN pip install --no-cache-dir ./services/memory-mcp

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "memory_mcp.server"]
