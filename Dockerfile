FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY app ./app
COPY models ./models

EXPOSE 9696

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9696"]