FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire workspace into the container
COPY . /app/

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Create virtual environment and install openenv + requirements
# Note: we use standard pip if uv falls through, but uv is faster
RUN cd smart_traffic && \
    uv venv /app/.venv && \
    VIRTUAL_ENV=/app/.venv uv pip install -e .[all] && \
    VIRTUAL_ENV=/app/.venv uv pip install fastapi uvicorn pydantic openai

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose standard HF container port
EXPOSE 8000

# Start the OpenEnv fastAPI server
CMD ["uvicorn", "smart_traffic.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
