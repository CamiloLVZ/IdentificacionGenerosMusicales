# Quickstart — API de Clasificación de Géneros Musicales

## Requisitos

- Python 3.10 o 3.11
- Git

## Pasos

### 1. Clonar el repositorio

```bash
git clone https://github.com/CamiloLVZ/IdentificacionGenerosMusicales.git
cd IdentificacionGenerosMusicales
```

### 2. Crear y activar el entorno virtual

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> La primera instalación puede tardar varios minutos por TensorFlow.

### 4. Levantar la API

```bash
cd backend
uvicorn main:app --reload
```

La API queda disponible en `http://localhost:8000`.

---

## Uso

### Documentación interactiva

Abre en el navegador:
```
http://localhost:8000/docs
```

### Predecir género de un audio (curl)

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@ruta/a/tu/audio.wav"
```

### Respuesta esperada

```json
{
  "predicted_genre": "rock",
  "confidence": 0.91,
  "probabilities": {
    "blues": 0.01,
    "rock": 0.91,
    "..."
  }
}
```

> El archivo debe ser `.wav` de máximo 30 segundos.
