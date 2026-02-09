# Lightweight Dockerfile for Render free tier (demo mode)
# No PyTorch/transformers — fits easily in 512MB
FROM python:3.11-slim

WORKDIR /app

# Install only minimal dependencies
COPY backend/requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy only the demo server
COPY backend/demo_server.py backend/demo_server.py

ENV DEMO_MODE=true

EXPOSE 8000

# Use shell form so $PORT is expanded (Render sets PORT dynamically)
CMD uvicorn backend.demo_server:app --host 0.0.0.0 --port ${PORT:-8000}
