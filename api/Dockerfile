# Dockerfile for FastAPI + ML Model
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Copy ML models and data
COPY models/ ./models/
COPY data/ ./data/

# Copy raw data if exists (optional, for full functionality)
COPY 2025_LoL_esports_match_data_from_OraclesElixir.csv* ./

# Expose port (Railway uses PORT env variable)
EXPOSE 8000

# Run the application - use PORT env variable if available
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
