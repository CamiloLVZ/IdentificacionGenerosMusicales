"""
API REST para clasificar generos musicales.

Ejecucion desde la carpeta backend:
    uvicorn main:app --host 0.0.0.0 --reload
"""

from fastapi import FastAPI, File, HTTPException, UploadFile

from predictor import load_model_and_labels, predict_genre


app = FastAPI(
    title="GTZAN Music Genre Classifier",
    description="API para clasificar generos musicales desde audios WAV.",
    version="1.0.0",
)


model = None
label_encoder = None
preprocessing_config = None


@app.on_event("startup")
def startup_event():
    """Carga el modelo una sola vez al iniciar la API."""
    global model, label_encoder, preprocessing_config
    model, label_encoder, preprocessing_config = load_model_and_labels()


@app.get("/")
def home():
    """Verifica que la API esta activa."""
    return {"message": "API de clasificacion de generos musicales funcionando"}


@app.get("/health")
def health():
    """
    Estado del servicio.

    Indica si el modelo esta cargado y listo para recibir peticiones.
    """
    ready = model is not None and label_encoder is not None
    return {"status": "ok" if ready else "error", "model_loaded": ready}


@app.get("/genres")
def genres():
    """
    Lista de generos que el modelo puede predecir.

    Util para que los clientes sepan que etiquetas puede devolver /predict.
    """
    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return {"genres": label_encoder.classes_.tolist()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe un archivo WAV y responde el genero predicho.

    Respuesta:
    {
      "predicted_genre": "rock",
      "confidence": 0.91,
      "probabilities": {"blues": 0.01, ...}
    }
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="El archivo debe ser WAV.")

    try:
        wav_bytes = await file.read()

        if len(wav_bytes) == 0:
            raise HTTPException(status_code=400, detail="El archivo esta vacio.")

        return predict_genre(model, label_encoder, preprocessing_config, wav_bytes)

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No fue posible procesar el audio: {error}",
        )
