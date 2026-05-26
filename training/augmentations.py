"""
Aumentos de datos simples para audio y espectrogramas.

Todos estos aumentos se usan SOLO en entrenamiento. La idea es que el modelo
vea pequenas variaciones de una misma cancion y aprenda patrones mas generales,
no detalles exactos de un archivo.
"""

import librosa
import numpy as np


def add_noise(audio, noise_factor=0.004):
    """Agrega ruido suave para simular grabaciones menos limpias."""
    noise = np.random.normal(0, 1, len(audio))
    return audio + noise_factor * noise


def shift_audio(audio, shift_max=0.15):
    """Mueve el audio un poco en el tiempo."""
    max_shift = int(len(audio) * shift_max)
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(audio, shift)


def pitch_shift(audio, sample_rate, max_steps=2):
    """Sube o baja ligeramente el tono del audio."""
    steps = np.random.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=steps)


def time_stretch(audio, min_rate=0.90, max_rate=1.10):
    """
    Cambia un poco la velocidad sin cambiar el tono.

    Si el audio queda mas corto o mas largo, el preprocesamiento lo corrige con
    padding o truncado.
    """
    rate = np.random.uniform(min_rate, max_rate)
    return librosa.effects.time_stretch(y=audio, rate=rate)


def random_volume(audio, min_gain=0.75, max_gain=1.25):
    """Sube o baja el volumen de forma moderada."""
    gain = np.random.uniform(min_gain, max_gain)
    return audio * gain


def random_audio_augmentation(audio, sample_rate):
    """
    Aplica un aumento elegido al azar.

    Mantener un solo aumento por copia evita crear ejemplos demasiado
    artificiales y mantiene el flujo facil de explicar.
    """
    augmentations = [
        lambda x: add_noise(x),
        lambda x: shift_audio(x),
        lambda x: pitch_shift(x, sample_rate),
        lambda x: time_stretch(x),
        lambda x: random_volume(x),
    ]
    selected_index = np.random.randint(0, len(augmentations))
    selected_augmentation = augmentations[selected_index]
    return selected_augmentation(audio)


def spec_augment(spectrogram, max_frequency_mask=12, max_time_mask=18):
    """
    Aplica SpecAugment simple sobre el espectrograma.

    Se oculta una pequena franja de frecuencias y una pequena franja de tiempo.
    Esto ayuda al modelo a no depender de una zona exacta del espectrograma.
    """
    augmented = np.array(spectrogram, copy=True)

    # El espectrograma llega con forma: frecuencias, tiempo, canal.
    frequency_bins = augmented.shape[0]
    time_steps = augmented.shape[1]

    freq_mask_size = np.random.randint(0, max_frequency_mask + 1)
    if freq_mask_size > 0 and freq_mask_size < frequency_bins:
        freq_start = np.random.randint(0, frequency_bins - freq_mask_size)
        augmented[freq_start : freq_start + freq_mask_size, :, :] = 0

    time_mask_size = np.random.randint(0, max_time_mask + 1)
    if time_mask_size > 0 and time_mask_size < time_steps:
        time_start = np.random.randint(0, time_steps - time_mask_size)
        augmented[:, time_start : time_start + time_mask_size, :] = 0

    return augmented
