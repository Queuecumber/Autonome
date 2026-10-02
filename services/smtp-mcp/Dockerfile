FROM python:3.12-slim
WORKDIR /app
COPY services/smtp-mcp/ ./services/smtp-mcp/
RUN pip install --no-cache-dir ./services/smtp-mcp
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "smtp_mcp.server"]
