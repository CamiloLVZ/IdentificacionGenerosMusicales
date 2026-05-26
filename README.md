# Clasificacion Automatica de Generos Musicales con GTZAN

Proyecto universitario de Computacion Inteligente para clasificar generos musicales a partir de archivos WAV.

El sistema usa Mel Spectrograms generados dinamicamente con `librosa` y modelos de aprendizaje profundo con TensorFlow/Keras. Tambien permite usar MFCCs como configuracion opcional. Incluye entrenamiento, API REST con FastAPI e interfaz grafica con Streamlit.

## Generos soportados

- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

## Estructura del proyecto

```text
project/
├── backend/
│   ├── main.py
│   ├── predictor.py
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
├── training/
│   ├── train.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── requirements.txt
├── models/
├── dataset/
├── docs/
│   ├── USER_GUIDE.md
│   └── TECHNICAL_DOCUMENTATION.md
├── outputs/
├── README.md
└── requirements.txt
```

## Preparar el dataset

Coloca GTZAN dentro de `dataset/` con una carpeta por genero:

```text
dataset/
├── blues/
│   ├── blues.00000.wav
│   └── ...
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

El archivo corrupto `jazz.00054.wav` se ignora automaticamente durante el entrenamiento.

## Instalacion

Desde la raiz del proyecto:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Entrenar los modelos

```bash
python training/train.py
```

El entrenamiento crea:

- `models/best_model.keras`
- `models/label_encoder.pkl`
- `models/preprocessing_config.json`
- `outputs/metrics/metrics.json`
- `outputs/metrics/cross_validation.json`
- graficas en `outputs/plots/`

Se entrenan dos modelos:

- CNN simple
- CNN + BiLSTM

El mejor se selecciona segun `validation accuracy`.

Por defecto, cada audio se divide en segmentos de 4 segundos con 50% de overlap. Esto aumenta la cantidad efectiva de ejemplos y suele mejorar la generalizacion.

La configuracion recomendada esta en:

```text
training/config.py
```

Puedes entrenar sin parametros por consola. Si quieres experimentar, cambia ahi valores como `FEATURE_TYPE`, `SEGMENT_DURATION`, `OVERLAP` o `BATCH_SIZE`.

Opciones utiles si aun quieres sobrescribir algo desde consola:

```bash
python training/train.py --feature_type mel
python training/train.py --feature_type mfcc_delta
python training/train.py --segment_duration 5 --overlap 0.5
python training/train.py --skip_cross_validation
```

`--skip_cross_validation` sirve para hacer una ejecucion mas rapida. Si no se usa, el script ejecuta validacion cruzada estratificada de 2 folds para el modelo CNN + BiLSTM.

## Ejecutar el backend

Desde la carpeta `backend`:

```bash
uvicorn main:app --reload
```

La API queda disponible en:

```text
http://127.0.0.1:8000
```

Endpoint principal:

```text
POST /predict
```

## Ejecutar el frontend

En otra terminal, desde la carpeta `frontend`:

```bash
streamlit run streamlit_app.py
```

Luego sube un archivo WAV y presiona el boton para predecir el genero.

## Flujo general

```text
Audio WAV -> Streamlit -> FastAPI -> Preprocesamiento -> Modelo -> Resultado
```

## Notas importantes

- Los espectrogramas se generan desde los audios, no desde imagenes pre-generadas.
- El aumento de datos se aplica solo al entrenamiento.
- La division train/validation/test se hace por archivo antes de segmentar para evitar fuga de datos.
- El codigo prioriza claridad sobre sofisticacion.
- El proyecto esta pensado para ejecutarse localmente en Windows.

## Documentacion para estudiar el proyecto

Documentacion general:

- `docs/README.md`
- `docs/PROJECT_WALKTHROUGH.md`
- `docs/TECHNICAL_DOCUMENTATION.md`
- `docs/USER_GUIDE.md`

Documentacion por carpeta:

- `training/README.md`
- `backend/README.md`
- `frontend/README.md`
