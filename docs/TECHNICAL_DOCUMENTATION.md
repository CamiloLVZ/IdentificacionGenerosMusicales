# Documentacion Tecnica

Este documento explica el proyecto de forma sencilla, pensando en estudiantes que deben entender y exponer el codigo.

## 1. Objetivo

El objetivo es clasificar archivos de audio WAV en uno de los 10 generos del dataset GTZAN.

El sistema completo tiene tres partes:

```text
Entrenamiento -> API REST -> Interfaz Streamlit
```

## 2. Preprocesamiento

Archivo principal:

```text
training/preprocessing.py
```

El modelo no recibe el audio crudo directamente. Primero convertimos el audio en un Mel Spectrogram.

### Pasos principales

1. Cargar el audio con `librosa.load`.
2. Convertirlo a mono.
3. Usar una frecuencia de muestreo fija de 22050 Hz.
4. Tomar 10 segundos del audio para reducir tiempo y memoria.
5. Generar Mel Spectrogram con `librosa.feature.melspectrogram`.
6. Pasar a decibelios con `librosa.power_to_db`.
7. Normalizar restando media y dividiendo por desviacion estandar.
8. Aplicar padding o truncado para que todos tengan el mismo tamano.
9. Agregar un canal final para que Keras lo trate como imagen.

Forma final:

```text
(128, 431, 1)
```

Esto significa:

- 128 bandas Mel.
- 431 pasos temporales.
- 1 canal.

## 3. Division del dataset

Archivo:

```text
training/train.py
```

La division es:

- 70% entrenamiento.
- 15% validacion.
- 15% prueba.

Se usa `stratify` para mantener una proporcion similar de generos en cada grupo.

## 4. Aumento de datos

El aumento se aplica solo a entrenamiento.

Por cada audio de entrenamiento se generan:

- original
- con ruido
- con desplazamiento temporal
- con pitch shifting

Esto ayuda a que el modelo no memorice exactamente los audios originales.

No se aumenta validacion ni prueba, porque esos conjuntos deben medir el rendimiento con datos reales.

## 5. Modelos

Archivo:

```text
training/models.py
```

### CNN simple

La CNN ve el Mel Spectrogram como una imagen. Aprende patrones de frecuencia y tiempo usando capas convolucionales.

Componentes:

- `Conv2D`
- `MaxPooling2D`
- `Dropout`
- `Flatten`
- `Dense`
- `Softmax`

### CNN + BiLSTM

Este modelo combina:

- CNN para extraer caracteristicas del espectrograma.
- BiLSTM para analizar la evolucion temporal de esas caracteristicas.

La idea es que la musica tiene cambios a lo largo del tiempo, no solo patrones estaticos.

## 6. Entrenamiento

Archivo:

```text
training/train.py
```

El script entrena los dos modelos y compara su `validation accuracy`.

Callbacks usados:

- `EarlyStopping`: detiene entrenamiento si no mejora.
- `ReduceLROnPlateau`: reduce el learning rate si el modelo se estanca.

Archivos generados:

```text
models/best_model.keras
models/label_encoder.pkl
outputs/metrics/metrics.json
outputs/plots/simple_cnn_history.png
outputs/plots/cnn_bilstm_history.png
outputs/plots/simple_cnn_confusion_matrix.png
outputs/plots/cnn_bilstm_confusion_matrix.png
```

## 7. Evaluacion

Archivo:

```text
training/evaluation.py
```

Metricas calculadas:

- accuracy
- precision
- recall
- F1-score
- matriz de confusion

La matriz de confusion permite ver que generos se confunden mas entre si.

## 8. API REST

Archivos:

```text
backend/main.py
backend/predictor.py
```

La API usa FastAPI.

Endpoint:

```text
POST /predict
```

La API:

1. Carga el modelo una sola vez al iniciar.
2. Recibe un archivo WAV.
3. Guarda temporalmente el archivo.
4. Aplica el mismo preprocesamiento usado en entrenamiento.
5. Ejecuta `model.predict`.
6. Devuelve genero, confianza y probabilidades.

Respuesta:

```json
{
  "predicted_genre": "rock",
  "confidence": 0.91,
  "probabilities": {
    "blues": 0.01,
    "classical": 0.00,
    "country": 0.02,
    "disco": 0.01,
    "hiphop": 0.01,
    "jazz": 0.00,
    "metal": 0.02,
    "pop": 0.01,
    "reggae": 0.01,
    "rock": 0.91
  }
}
```

## 9. Frontend

Archivo:

```text
frontend/streamlit_app.py
```

Streamlit permite:

- subir un WAV
- reproducirlo
- enviarlo a la API
- mostrar genero predicho
- mostrar probabilidades
- dibujar grafico de barras

El frontend no carga TensorFlow ni el modelo. Solo consume la API.

## 10. Flujo completo

```text
Usuario sube WAV
        |
        v
Streamlit envia archivo a FastAPI
        |
        v
FastAPI llama predictor.py
        |
        v
preprocessing.py genera Mel Spectrogram
        |
        v
TensorFlow predice probabilidades
        |
        v
Streamlit muestra resultados
```

## 11. Por que el codigo es simple

El proyecto evita clases y arquitecturas empresariales porque el objetivo es educativo.

Se usan funciones pequeñas y archivos separados por responsabilidad:

- `preprocessing.py`: preparar audios.
- `models.py`: crear modelos.
- `evaluation.py`: medir resultados.
- `train.py`: flujo completo de entrenamiento.
- `predictor.py`: logica de prediccion.
- `main.py`: API.
- `streamlit_app.py`: interfaz.

