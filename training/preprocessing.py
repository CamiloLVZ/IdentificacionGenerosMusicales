"""
Funciones de preprocesamiento para el proyecto GTZAN.

Este archivo contiene todo lo necesario para leer audios WAV y convertirlos en
Mel Spectrograms. Se usa tanto durante entrenamiento como durante prediccion,
porque el modelo debe recibir los datos siempre con el mismo formato.
"""

from pathlib import Path

import librosa
import numpy as np


# Semilla fija para que los experimentos sean mas reproducibles.
RANDOM_SEED = 42

# GTZAN normalmente viene con audios de 30 segundos, pero para entrenamiento
# usamos los primeros 10 segundos. Esto reduce memoria y tiempo de entrenamiento,
# manteniendo suficiente informacion musical para un proyecto universitario.
SAMPLE_RATE = 22050
DURATION_SECONDS = 10
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Cantidad esperada de frames de tiempo para 10 segundos de audio.
EXPECTED_TIME_STEPS = 431

# Generos oficiales del dataset GTZAN.
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def load_audio(file_path, sample_rate=SAMPLE_RATE, duration=DURATION_SECONDS):
    """
    Carga un archivo WAV con librosa.

    librosa convierte el audio a mono por defecto. Esto simplifica el proyecto,
    porque todos los audios quedan representados como una sola senal.
    """
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration, mono=True)
    return audio, sr


def add_noise(audio, noise_factor=0.005):
    """Agrega ruido suave al audio para hacer el modelo un poco mas robusto."""
    noise = np.random.normal(0, 1, len(audio))
    return audio + noise_factor * noise


def shift_audio(audio, shift_max=0.2):
    """
    Desplaza el audio en el tiempo.

    np.roll mueve la senal hacia adelante o hacia atras. Esto simula que una
    cancion empieza un poco antes o despues.
    """
    max_shift = int(len(audio) * shift_max)
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(audio, shift)


def pitch_shift(audio, sample_rate=SAMPLE_RATE, n_steps=2):
    """
    Cambia ligeramente el tono del audio.

    Este aumento se usa solo en entrenamiento. No se aplica en validacion,
    prueba ni prediccion real.
    """
    steps = np.random.uniform(-n_steps, n_steps)
    return librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=steps)


def audio_to_mel_spectrogram(audio, sample_rate=SAMPLE_RATE):
    """
    Convierte audio en Mel Spectrogram normalizado.

    Pasos:
    1. Calcula el Mel Spectrogram.
    2. Lo pasa a escala decibelios, mas parecida a como percibimos el sonido.
    3. Normaliza con media 0 y desviacion estandar 1.
    4. Ajusta el tamano con padding o truncado.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = normalize_spectrogram(mel_db)
    mel_db = pad_or_truncate(mel_db, EXPECTED_TIME_STEPS)

    # Keras espera imagenes con canal: alto, ancho, canales.
    return mel_db[..., np.newaxis].astype(np.float32)


def normalize_spectrogram(spectrogram):
    """
    Normalizacion simple.

    Se resta la media y se divide por la desviacion estandar. El valor pequeno
    evita divisiones por cero si algun audio raro tiene desviacion 0.
    """
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    return (spectrogram - mean) / (std + 1e-8)


def pad_or_truncate(spectrogram, target_time_steps):
    """
    Ajusta todos los espectrogramas al mismo ancho temporal.

    Las redes neuronales necesitan entradas del mismo tamano. Si el audio queda
    corto, agregamos ceros. Si queda largo, recortamos.
    """
    current_time_steps = spectrogram.shape[1]

    if current_time_steps < target_time_steps:
        missing_steps = target_time_steps - current_time_steps
        spectrogram = np.pad(
            spectrogram,
            pad_width=((0, 0), (0, missing_steps)),
            mode="constant",
        )
    elif current_time_steps > target_time_steps:
        spectrogram = spectrogram[:, :target_time_steps]

    return spectrogram


def process_audio_file(file_path):
    """Carga un WAV y lo convierte al formato que espera el modelo."""
    audio, sr = load_audio(file_path)
    return audio_to_mel_spectrogram(audio, sr)


def get_dataset_files(dataset_path):
    """
    Recorre la carpeta dataset y devuelve rutas de audio con sus etiquetas.

    Estructura esperada:
    dataset/
      blues/blues.00000.wav
      classical/classical.00000.wav
      ...
    """
    dataset_path = Path(dataset_path)
    audio_files = []
    labels = []

    for genre in GENRES:
        genre_folder = dataset_path / genre
        if not genre_folder.exists():
            print(f"Advertencia: no existe la carpeta {genre_folder}")
            continue

        for wav_file in sorted(genre_folder.glob("*.wav")):
            # Este archivo de GTZAN es conocido por estar corrupto.
            if wav_file.name == "jazz.00054.wav":
                print("Ignorando archivo corrupto: jazz.00054.wav")
                continue

            audio_files.append(str(wav_file))
            labels.append(genre)

    return audio_files, labels


def create_augmented_examples(file_path):
    """
    Crea ejemplos aumentados para un audio de entrenamiento.

    Devuelve una lista con:
    - audio original
    - audio con ruido
    - audio desplazado
    - audio con pitch shifting
    """
    audio, sr = load_audio(file_path)

    augmented_audio = [
        audio,
        add_noise(audio),
        shift_audio(audio),
        pitch_shift(audio, sr),
    ]

    return [audio_to_mel_spectrogram(example, sr) for example in augmented_audio]
