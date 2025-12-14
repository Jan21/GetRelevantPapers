FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY classifiers/ ./classifiers/
COPY config.yaml ./

# Create necessary directories
RUN mkdir -p markdown_db/raw markdown_db/sections vector_store converted_papers outputs

# Expose port for web UI
EXPOSE 3444

# Default command
CMD ["python", "minimal_web_ui.py"]




