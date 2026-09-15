FROM python:3.12-slim
WORKDIR /app
COPY services/imap-mcp/ ./services/imap-mcp/
RUN pip install --no-cache-dir ./services/imap-mcp
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "imap_mcp.server"]
