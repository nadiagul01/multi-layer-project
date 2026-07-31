FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY app/ ./app/
COPY src/ ./src/
RUN mkdir -p app/outputs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
