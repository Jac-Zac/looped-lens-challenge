# The runner image: the competition environment, the runner, and the inputs of
# the eval datasets. Built from build/runner/, which `manage.py build` assembles
# and which never contains a dataset's private directory.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

COPY runner ./runner
COPY trust_proxy.py ./
COPY data /data

# The method runs as an ordinary user with a home directory of its own.
RUN useradd -m -u 1000 runner
USER runner
# Hugging Face's xet client does its own networking and ignores the proxy.
ENV HF_HUB_DISABLE_XET=1
CMD ["python", "trust_proxy.py"]
