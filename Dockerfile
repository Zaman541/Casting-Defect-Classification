FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY src/ src/
COPY api/ api/
COPY outputs/ outputs/

# Expose port
EXPOSE 8000

# Environment variables (override at runtime)
ENV MODEL_NAME=efficientnet_b0
ENV OUTPUT_DIR=outputs
ENV PORT=8000

# Start the API server
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
