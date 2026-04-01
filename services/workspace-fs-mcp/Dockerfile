FROM python:3.12-slim

WORKDIR /app

COPY services/workspace-fs-mcp/ ./services/workspace-fs-mcp/

RUN pip install --no-cache-dir ./services/workspace-fs-mcp

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "workspace_fs.server"]
