FROM python:3.12-slim

WORKDIR /app

COPY services/session-manager/ ./services/session-manager/

RUN pip install --no-cache-dir ./services/session-manager

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "session_manager.main"]
