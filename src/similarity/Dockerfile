FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/similarity ./src/similarity

# TODO убрать --extra dev, пока что он нужен для работы без pydantic-settings
# Устанавливаем базовые зависимости и группу similarity
RUN uv sync --frozen --extra similarity --extra dev

ENV PATH="/app/.venv/bin:$PATH"
