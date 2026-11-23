# The Hugging Face Space. `manage.py deploy` uploads this file, the lockfile,
# competition.py and space/ to the Space, which builds it.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

# Spaces run as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

# Only the Space's own packages: none of the competition environment (torch,
# nnsight) is needed to accept submissions and draw a table.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group space --no-cache

COPY competition.py ./
COPY space ./space

USER user
EXPOSE 7860
CMD ["uvicorn", "space.app:from_environment", "--factory", "--host", "0.0.0.0", "--port", "7860"]
