# Documentacion de la Carpeta `frontend`

Esta carpeta contiene la interfaz grafica con Streamlit.

El frontend no carga el modelo. Solo envia el audio al backend.

## Archivo Principal

```text
frontend/streamlit_app.py
```

## Flujo

```text
Usuario sube WAV
    |
Streamlit reproduce audio
    |
Usuario presiona "Predecir genero"
    |
Streamlit envia POST /predict
    |
FastAPI responde JSON
    |
Streamlit muestra resultado
```

## Variables

```python
API_URL = "http://127.0.0.1:8000/predict"
```

Indica donde esta el endpoint de prediccion.

Si cambias el puerto del backend, tambien debes cambiar esta URL.

## Componentes Streamlit

### `st.set_page_config`

Configura titulo y layout de la pagina.

### `st.file_uploader`

Permite subir archivos `.wav`.

```python
uploaded_file = st.file_uploader("Selecciona un audio WAV", type=["wav"])
```

### `st.audio`

Permite reproducir el audio subido.

### `st.button`

El usuario decide cuando enviar el audio a la API.

### `requests.post`

Envia el archivo al backend.

```python
response = requests.post(API_URL, files=files, timeout=60)
```

### `st.metric`

Muestra el genero predicho y la confianza.

### `st.bar_chart`

Muestra probabilidades por genero.

### `st.dataframe`

Muestra los valores numericos exactos.

## Errores Controlados

Si el backend no esta encendido:

```text
No se pudo conectar con la API.
```

Si la API tarda demasiado:

```text
La API tardo demasiado en responder.
```

Si la API responde error:

```text
Error en la API: ...
```

## Ejecutar Frontend

Desde `frontend/`:

```bash
streamlit run streamlit_app.py
```

Antes de usarlo, el backend debe estar ejecutandose.

