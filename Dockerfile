FROM python:3.9.16-slim-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

ENV PYTHONPATH=/app

COPY pyproject.toml .

RUN pip install -U pip setuptools wheel
RUN pip install pdm

RUN pdm install

ENTRYPOINT ["pdm", "run", "src/server.py"]
