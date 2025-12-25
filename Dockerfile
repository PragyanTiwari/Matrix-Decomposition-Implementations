# uv base image
FROM python:3.11-slim-bookworm

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the marimo notebook file
COPY apps/ notebooks/

# env. variable to define notebook to run (will be overriden by render at runtime)
ENV MARIMO_NOTEBOOK="notebooks/gram_schmidt_process.py" 

# Expose the marimo port
EXPOSE 2718

# Run the notebook through shell command
CMD ["sh", "-c", "uv run marimo run ${MARIMO_NOTEBOOK:?Set MARIMO_NOTEBOOK} --host 0.0.0.0 --port 2718"]


# Advancements:
# 3. make it well-define
