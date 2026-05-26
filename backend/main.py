"""
API REST sencilla para clasificar generos musicales.

Ejecucion desde la carpeta backend:
uvicorn main:app --reload
"""

from fastapi import FastAPI, File, HTTPException, UploadFile

from predictor import load_model_and_labels, predict_genre


app = FastAPI(
    title="GTZAN Music Genre Classifier",
    description="API educativa para clasificar generos musicales desde audios WAV.",
    version="1.0.0",
)


model = None
label_encoder = None
preprocessing_config = None


@app.on_event("startup")
def startup_event():
    """
    Carga el modelo una sola vez al iniciar la API.

    Si el modelo no existe, la API avisara con un error claro.
    """
    global model, label_encoder, preprocessing_config
    model, label_encoder, preprocessing_config = load_model_and_labels()


@app.get("/")
def home():
    """Endpoint simple para verificar que la API esta viva."""
    return {"message": "API de clasificacion de generos musicales funcionando"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe un archivo WAV y responde el genero predicho.

    Respuesta esperada:
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
