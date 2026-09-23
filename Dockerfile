FROM python:3.11-slim

WORKDIR /app

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specifications first for layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application code
COPY . .

# Final sync to include project packages
RUN uv sync --frozen --no-dev

# Expose port 8000
EXPOSE 8000

# Run uvicorn on 0.0.0.0 so port publishing (-p) works
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

