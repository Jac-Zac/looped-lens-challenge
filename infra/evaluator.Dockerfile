# The evaluator image: the competition environment plus the `evaluator`
# dependency group, the scoring code with its judge (final-evals/), and the eval
# datasets with their private directories. Built from build/evaluator/, which
# `manage.py build` assembles.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --group evaluator

COPY evaluator ./evaluator
COPY final-evals ./final-evals
COPY data /data

RUN useradd -m -u 1000 evaluator
USER evaluator
CMD ["python", "-m", "evaluator.task"]
