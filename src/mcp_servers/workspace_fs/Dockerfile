FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastmcp>=2.0

COPY server.py .

ENV WORKSPACE_DIR=/workspace
ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
