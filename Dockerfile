FROM python:3.11-slim

# libsndfile1 es requerido por librosa para leer archivos WAV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias primero (aprovecha la cache de Docker)
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y los artefactos necesarios
COPY backend/ backend/
COPY training/preprocessing.py training/preprocessing.py
COPY models/ models/

EXPOSE 8000

# Ejecutar desde /app/backend para que los imports relativos funcionen correctamente
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

