FROM python:3.12-slim

# ---- System deps (needed for numpy / faiss) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Install Python deps first (better caching) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY app ./app

# ---- Create storage dir for FAISS ----
RUN mkdir -p /app/storage

# ---- Expose API port ----
EXPOSE 8000

# ---- Run FastAPI ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
