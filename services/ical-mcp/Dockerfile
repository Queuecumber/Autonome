FROM python:3.12-slim
WORKDIR /app
COPY services/ical-mcp/ ./services/ical-mcp/
RUN pip install --no-cache-dir ./services/ical-mcp
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "ical_mcp.server"]
