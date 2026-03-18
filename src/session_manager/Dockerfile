FROM python:3.12-slim

WORKDIR /app

COPY src/session_manager/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "session_manager.main"]
