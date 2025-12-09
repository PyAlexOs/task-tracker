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
COPY src/text_summarizer ./src/text_summarizer

# TODO убрать --extra dev, пока что он нужен для работы без pydantic-settings
# Устанавливаем базовые зависимости и группу text-summarizer
RUN uv sync --frozen --extra text-summarizer --extra dev

ENV PATH="/app/.venv/bin:$PATH"
