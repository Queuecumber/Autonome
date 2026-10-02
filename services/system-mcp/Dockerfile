FROM python:3.12-slim

WORKDIR /app

COPY services/system-mcp/ ./services/system-mcp/

RUN pip install --no-cache-dir ./services/system-mcp

# Fail the image build if dependencies don't resolve into an importable server
RUN python -c "import system_mcp.server"

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "system_mcp.server"]
