"""
Interfaz grafica con Streamlit.

Esta app NO carga el modelo directamente. Consume la API FastAPI para respetar
el flujo pedido:

Streamlit -> FastAPI -> Modelo
"""

import pandas as pd
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="Clasificador de Generos Musicales",
    layout="centered",
)

st.title("Clasificador de Generos Musicales")
st.write("Sube un archivo WAV y el sistema intentara predecir su genero musical.")

uploaded_file = st.file_uploader("Selecciona un audio WAV", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predecir genero"):
        with st.spinner("Enviando audio a la API..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "audio/wav",
                )
            }

            try:
                response = requests.post(API_URL, files=files, timeout=60)

                if response.status_code != 200:
                    st.error(f"Error en la API: {response.text}")
                else:
                    result = response.json()

                    st.subheader("Resultado")
                    st.metric(
                        label="Genero predicho",
                        value=result["predicted_genre"],
                        delta=f"{result['confidence']:.2%} confianza",
                    )

                    st.subheader("Probabilidades por genero")
                    probabilities = result["probabilities"]
                    chart_data = pd.DataFrame(
                        {
                            "genero": list(probabilities.keys()),
                            "probabilidad": list(probabilities.values()),
                        }
                    ).set_index("genero")

                    st.bar_chart(chart_data)
                    st.dataframe(chart_data, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error(
                    "No se pudo conectar con la API. Verifica que FastAPI este ejecutandose."
                )
            except requests.exceptions.Timeout:
                st.error("La API tardo demasiado en responder.")
