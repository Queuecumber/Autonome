FROM python:3.12-slim

WORKDIR /app

COPY services/graphiti-mcp/ ./services/graphiti-mcp/

RUN pip install --no-cache-dir ./services/graphiti-mcp

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "graphiti_mcp.server"]
