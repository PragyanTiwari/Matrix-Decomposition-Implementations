FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /usr/local/bin/uv /usr/local/bin/

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --system

# Copy app
COPY apps/ .

# Default notebook
ENV MARIMO_NOTEBOOK=gs_process.py

EXPOSE 2718

CMD ["sh", "-c", "uv run marimo run ${MARIMO_NOTEBOOK:-gs_process.py} --host 0.0.0.0 --port 2718"]