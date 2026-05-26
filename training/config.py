"""
Configuracion central del entrenamiento.

La idea es que el estudiante no tenga que recordar muchos parametros por
consola. Si quiere experimentar, puede cambiar estos valores aqui.
"""


# Ruta por defecto del dataset. Debe contener las carpetas blues, classical, etc.
DATASET_PATH = "dataset"

# Carpetas de salida.
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

# Configuracion recomendada para mejorar el rendimiento sin complicar el modelo.
FEATURE_TYPE = "mel"
SEGMENT_DURATION = 4.0
OVERLAP = 0.5
AUGMENTATIONS_PER_SEGMENT = 1

# Entrenamiento.
EPOCHS = 60
CV_EPOCHS = 20
BATCH_SIZE = 32

# Por defecto dejamos la validacion cruzada activa porque era requisito.
# Si necesitas una prueba rapida, cambia este valor a True.
SKIP_CROSS_VALIDATION = False

