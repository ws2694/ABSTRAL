FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (not data — that lives on the persistent volume)
COPY . .

# Port
EXPOSE 8420

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8420/api/status')"

# Run
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8420"]
