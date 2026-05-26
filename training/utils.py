"""
Utilidades pequenas compartidas por el entrenamiento.
"""

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_random_seed(seed):
    """Fija semillas para que los experimentos sean mas reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(data, file_path):
    """Guarda datos en JSON con formato legible."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

