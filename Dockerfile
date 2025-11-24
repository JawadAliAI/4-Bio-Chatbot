FROM python:3.10-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories for data persistence
# Note: In Render free tier, filesystem is ephemeral. 
# For persistence, you'd need a Render Disk (paid) or external DB.
RUN mkdir -p patient_histories sessions TTS_Output

# Expose the port
EXPOSE 8000

# Start the application
# Using shell form to allow variable expansion if needed, but explicit port is safer for now
# Render sets PORT environment variable, so we use it.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
