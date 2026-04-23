# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for scikit-learn/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p saved_models results data/raw

# Expose the Flask port
EXPOSE 5001

# Set Environment Variables
ENV FLASK_APP=app.routes
ENV PYTHONIOENCODING=utf-8

# Start the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5001"]
