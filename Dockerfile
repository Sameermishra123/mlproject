# Use official Python 3.8 slim image
FROM python:3.8-slim

# Avoid creating .pyc files and force stdout/stderr logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /application

# Install system dependencies needed for scientific stack
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all code first (required for -e . installs)
COPY . .

# Upgrade pip and install Python dependencies with longer timeout & retries
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=10 -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "application.py"]
