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
uvicorn main:app --host 0.0.0.0 --reload
```

- `--host 0.0.0.0` expone la API en todas las interfaces de red, lo que permite acceder al servicio desde otros equipos de la misma red local.
- La API queda disponible en `http://<IP-del-servidor>:8000`.

> Para conocer tu IP local ejecuta `ipconfig` (Windows) o `ip a` (Linux/macOS).

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Verifica que la API está activa |
| `GET` | `/health` | Estado del servicio y del modelo |
| `GET` | `/genres` | Lista de géneros que el modelo puede predecir |
| `POST` | `/predict` | Clasifica un audio WAV |

### Documentación interactiva

```
http://<IP-del-servidor>:8000/docs
```

### Ejemplos con curl

```bash
# Estado del servicio
curl http://localhost:8000/health

# Géneros disponibles
curl http://localhost:8000/genres

# Predecir género de un audio
curl -X POST "http://localhost:8000/predict" \
     -F "file=@ruta/a/tu/audio.wav"
```

### Respuesta de /predict

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

