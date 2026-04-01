FROM python:3.12-slim

WORKDIR /app

COPY lib/agent-platform-core/ ./lib/agent-platform-core/
COPY services/session-manager/ ./services/session-manager/

RUN pip install --no-cache-dir \
    ./lib/agent-platform-core \
    ./services/session-manager

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "session_manager.main"]
