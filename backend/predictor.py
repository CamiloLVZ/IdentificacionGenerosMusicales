"""
Logica de prediccion para la API.

El modelo se carga una sola vez cuando inicia FastAPI. Luego se reutiliza en
cada peticion, lo cual es mas rapido que cargarlo por cada archivo.
"""

import pickle
import sys
import tempfile
from pathlib import Path
import json

import numpy as np
import tensorflow as tf


# Permitimos importar el preprocesamiento desde la carpeta training.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = PROJECT_ROOT / "training"
sys.path.append(str(TRAINING_PATH))

from preprocessing import process_audio_file  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
PREPROCESSING_CONFIG_PATH = PROJECT_ROOT / "models" / "preprocessing_config.json"


def load_model_and_labels():
    """Carga el modelo entrenado y el codificador de etiquetas."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {MODEL_PATH}. Primero ejecuta training/train.py."
        )

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro label_encoder.pkl en {LABEL_ENCODER_PATH}."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)

    preprocessing_config = {
        "feature_type": "mel",
        # Valores compatibles con el primer modelo del proyecto. Cuando se
        # reentrena con la nueva version, train.py guarda esta configuracion.
        "segment_duration": 10.0,
        "overlap": 0.0,
    }

    if PREPROCESSING_CONFIG_PATH.exists():
        with open(PREPROCESSING_CONFIG_PATH, "r", encoding="utf-8") as file:
            preprocessing_config.update(json.load(file))

    return model, label_encoder, preprocessing_config


def predict_genre(model, label_encoder, preprocessing_config, wav_bytes):
    """
    Predice el genero a partir de bytes de un archivo WAV.

    FastAPI recibe el archivo en memoria. Como librosa trabaja comodamente con
    rutas de archivos, guardamos temporalmente el WAV y luego lo procesamos.
    """
    # En Windows no conviene usar NamedTemporaryFile(delete=True) para este caso,
    # porque el archivo queda abierto y librosa no puede leerlo de nuevo.
    # Por eso lo creamos, lo cerramos, lo leemos y luego lo borramos manualmente.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(wav_bytes)
            temp_path = temp_file.name

        segments = process_audio_file(
            temp_path,
            feature_type=preprocessing_config["feature_type"],
            segment_duration=preprocessing_config["segment_duration"],
            overlap=preprocessing_config["overlap"],
        )

    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

    # El modelo predice cada segmento. Luego promediamos las probabilidades para
    # obtener una sola respuesta para el audio completo.
    segment_probabilities = model.predict(segments)
    probabilities = np.mean(segment_probabilities, axis=0)

    predicted_index = int(np.argmax(probabilities))
    predicted_genre = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])

    probabilities_by_genre = {
        genre: float(probabilities[index])
        for index, genre in enumerate(label_encoder.classes_)
    }

    return {
        "predicted_genre": predicted_genre,
        "confidence": confidence,
        "probabilities": probabilities_by_genre,
    }
