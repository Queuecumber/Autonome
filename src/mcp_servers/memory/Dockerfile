FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastmcp>=2.0

COPY server.py .

ENV MEMORY_DIR=/workspace/memory
ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
