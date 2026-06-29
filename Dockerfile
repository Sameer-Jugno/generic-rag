# Phase 1: Environment Base Setup
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer streams for instant logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Phase 2: Dependency Caching Layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Phase 3: Code Injection & Permissions Setup
COPY . .

# Grant executable permissions to our multi-process manager script
RUN chmod +x run.sh

# Expose the Streamlit web layout interface port
EXPOSE 7860

# Execute the orchestrator script on system boot
CMD ["./run.sh"]
