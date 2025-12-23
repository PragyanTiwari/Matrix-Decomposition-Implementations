FROM python:3.13-slim-bookworm

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the marimo notebook file
COPY gram_schmidt_process.py .

# Expose the marimo port
EXPOSE 2718

# Run the notebook
CMD ["uv", "run", "marimo", "run", "gram_schmidt_process.py", "--host", "0.0.0.0", "--port", "2718"]


# Advancements:
# 1. adding env variable to CMD
# 2. multiple notebooks integration
# 3. make it well-define