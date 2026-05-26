# Documentacion de la Carpeta `training`

Esta carpeta contiene todo el codigo necesario para entrenar y evaluar los modelos.

## Archivos

```text
training/
├── config.py
├── preprocessing.py
├── augmentations.py
├── models.py
├── evaluation.py
├── train.py
├── utils.py
└── requirements.txt
```

## 1. `config.py`

Archivo de configuracion central.

Sirve para no tener que pasar muchos parametros por consola.

Variables principales:

```python
DATASET_PATH = "dataset"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
```

Indican donde esta el dataset y donde se guardan resultados.

```python
FEATURE_TYPE = "mel"
SEGMENT_DURATION = 4.0
OVERLAP = 0.5
```

Controlan el preprocesamiento.

- `FEATURE_TYPE`: tipo de representacion. Puede ser `mel`, `mfcc` o `mfcc_delta`.
- `SEGMENT_DURATION`: segundos por segmento.
- `OVERLAP`: porcentaje de solapamiento entre segmentos.

```python
AUGMENTATIONS_PER_SEGMENT = 1
```

Indica cuantas copias aumentadas se crean por segmento original.

```python
EPOCHS = 60
CV_EPOCHS = 20
BATCH_SIZE = 32
```

Controlan el entrenamiento.

```python
SKIP_CROSS_VALIDATION = False
```

Si esta en `False`, se ejecuta validacion cruzada de 2 folds.

## 2. `preprocessing.py`

Este archivo prepara los datos de audio.

### Constantes importantes

```python
SAMPLE_RATE = 22050
```

Todos los audios se convierten a 22050 Hz. Esto hace que el modelo trabaje con una frecuencia uniforme.

```python
MAX_AUDIO_DURATION_SECONDS = 30
```

Se leen maximo 30 segundos, que corresponde al formato comun de GTZAN.

```python
N_MELS = 128
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
```

Parametros de extraccion de features.

- `N_MELS`: numero de bandas Mel.
- `N_MFCC`: numero de coeficientes MFCC.
- `N_FFT`: tamano de la ventana para analizar frecuencias.
- `HOP_LENGTH`: salto entre ventanas.

### `get_expected_time_steps`

Calcula cuantos pasos de tiempo tendra cada segmento despues de convertirlo a espectrograma o MFCC.

Esto es importante porque Keras necesita entradas de tamano fijo.

### `get_feature_height`

Devuelve la altura de la matriz de entrada:

- `mel`: 128 filas
- `mfcc`: 20 filas
- `mfcc_delta`: 60 filas, porque concatena MFCC + delta + delta-delta

### `get_input_shape`

Devuelve la forma final que espera Keras:

```text
(alto, tiempo, canales)
```

Ejemplo con Mel:

```text
(128, tiempo, 1)
```

El canal final se agrega porque Keras trata la entrada como una imagen.

### `load_audio`

Carga un WAV usando `librosa.load`.

Hace tres cosas importantes:

- convierte el audio a mono
- lo remuestrea a 22050 Hz
- limita la duracion maxima

### `split_audio_into_segments`

Divide el audio en segmentos pequenos.

Ejemplo:

```text
segment_duration = 4.0
overlap = 0.5
```

Cada segmento dura 4 segundos y el siguiente empieza 2 segundos despues.

Esto aumenta la cantidad de muestras y permite que el modelo aprenda patrones locales.

### `pad_audio`

Si un audio o segmento es mas corto de lo esperado, agrega ceros al final.

Esto evita errores por tamanos variables.

### `audio_to_features`

Convierte un segmento de audio en:

- Mel Spectrogram
- MFCC
- MFCC + delta + delta-delta

Luego normaliza y ajusta el tamano.

### `normalize_features`

Aplica normalizacion por muestra:

```text
(features - media) / desviacion_estandar
```

Esto hace que los valores sean mas estables para el entrenamiento.

### `pad_or_truncate`

Asegura que todas las matrices tengan el mismo ancho temporal.

- si falta tiempo, agrega ceros
- si sobra tiempo, recorta

### `process_audio_file`

Funcion de alto nivel.

Recibe una ruta WAV y devuelve todos sus segmentos ya convertidos a features.

Se usa tanto en entrenamiento como en prediccion.

### `get_dataset_files`

Recorre el dataset y devuelve:

```python
audio_files, labels
```

Tambien ignora automaticamente:

```text
jazz.00054.wav
```

porque es conocido como archivo corrupto en GTZAN.

## 3. `augmentations.py`

Contiene aumentos de datos aplicados solo al entrenamiento.

### `add_noise`

Agrega ruido suave. Ayuda a que el modelo no dependa de una grabacion perfecta.

### `shift_audio`

Desplaza el audio en el tiempo. Simula que el fragmento empieza un poco antes o despues.

### `pitch_shift`

Cambia ligeramente el tono.

### `time_stretch`

Cambia un poco la velocidad sin cambiar el tono.

### `random_volume`

Sube o baja el volumen.

### `random_audio_augmentation`

Escoge una augmentation al azar.

El proyecto usa una augmentation por copia para mantener los ejemplos realistas.

### `spec_augment`

Oculta una pequena franja de tiempo y una pequena franja de frecuencia en el espectrograma.

Esto ayuda a que el modelo no dependa de una zona exacta.

## 4. `models.py`

Define las arquitecturas.

### `build_simple_cnn`

Modelo CNN simple.

Capas principales:

- `Conv2D`: detecta patrones locales
- `BatchNormalization`: estabiliza valores internos
- `MaxPooling2D`: reduce dimensiones
- `Dropout`: reduce sobreajuste
- `GlobalAveragePooling2D`: resume mapas de caracteristicas
- `Dense`: clasifica
- `Softmax`: produce probabilidades

### `build_cnn_bilstm`

Modelo principal recomendado.

Primero usa CNN para extraer patrones del espectrograma. Luego convierte la salida en una secuencia y usa BiLSTM para analizar el cambio temporal.

Este modelo suele ser mejor para audio porque la musica evoluciona con el tiempo.

## 5. `evaluation.py`

Evalua modelos y guarda graficas.

### `evaluate_model`

Evalua directamente arrays `x_test`, `y_test`.

Queda disponible como utilidad general.

### `evaluate_model_on_files`

Evalua por archivo completo.

Como cada archivo se divide en segmentos, la funcion:

1. procesa el archivo
2. predice cada segmento
3. promedia probabilidades
4. calcula metricas

Esto se parece al uso real en la API.

### `plot_training_history`

Guarda una imagen con:

- accuracy entrenamiento
- accuracy validacion
- loss entrenamiento
- loss validacion

### `plot_confusion_matrix`

Guarda la matriz de confusion.

Sirve para ver que generos se confunden.

## 6. `utils.py`

Funciones pequenas:

### `set_random_seed`

Fija semillas para reproducibilidad.

### `save_json`

Guarda diccionarios en archivos JSON legibles.

## 7. `train.py`

Archivo principal de entrenamiento.

### `load_features_without_augmentation`

Procesa archivos para validacion o prueba.

No aplica augmentation porque validacion y prueba deben representar datos reales.

### `load_training_features`

Procesa archivos de entrenamiento.

Por cada segmento:

1. guarda el segmento original
2. crea copias aumentadas
3. convierte todo a features

### `split_dataset`

Divide:

```text
70% entrenamiento
15% validacion
15% prueba
```

Usa `stratify` para mantener proporcion similar de generos.

### `train_one_model`

Entrena un modelo Keras.

Usa:

- `EarlyStopping`
- `ReduceLROnPlateau`
- `ModelCheckpoint`

### `run_cross_validation`

Ejecuta validacion cruzada estratificada de 2 folds para CNN + BiLSTM.

Esto mide estabilidad del modelo en diferentes particiones.

### `main`

Orquesta todo:

1. fija semilla
2. lee dataset
3. divide datos
4. codifica labels
5. ejecuta validacion cruzada
6. prepara train y validation
7. entrena modelos
8. evalua en test
9. guarda modelo, encoder, configuracion y metricas

## 8. Comando Principal

Desde la raiz del proyecto:

```bash
python training/train.py
```

Los parametros vienen de `training/config.py`.

