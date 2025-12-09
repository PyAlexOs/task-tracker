FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/api_gateway ./src/api_gateway

# TODO убрать --extra dev, пока что он нужен для работы без pydantic-settings
# Устанавливаем базовые зависимости и группу api-gateway
RUN uv sync --frozen --extra api-gateway --extra dev

ENV PATH="/app/.venv/bin:$PATH"
