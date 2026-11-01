import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "modelo_corazon.joblib"

st.set_page_config(page_title="ACA 1 - Clasificación cardíaca", page_icon="❤️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("❤️ Clasificación de datos asociados a enfermedades cardíacas")
st.caption("Proyecto académico de Aprendizaje Automático — ACA 1")

st.warning(
    "Prototipo educativo. No es un dispositivo médico, no diagnostica "
    "enfermedades y no debe utilizarse para tomar decisiones clínicas."
)

if not MODEL_PATH.exists():
    st.error("No se encuentra modelo_corazon.joblib. Ejecuta primero train_model.py.")
    st.stop()

model = load_model()

st.write(
    "Ingresa los valores de las variables del conjunto de datos para obtener "
    "la categoría estimada por el modelo de Regresión Logística."
)

with st.form("patient_form"):
    st.subheader("Datos de entrada")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad (años)", min_value=1, max_value=100, value=50)
        sex = st.selectbox("Sexo (dataset)", options=[0, 1],
                           format_func=lambda x: "0" if x == 0 else "1")
        cp = st.selectbox("Tipo de dolor torácico (cp)", [1, 2, 3, 4])
        trestbps = st.number_input("Presión arterial en reposo (trestbps)", 50, 250, 120)
        chol = st.number_input("Colesterol (chol)", 50, 700, 200)
        fbs = st.selectbox("Glucemia en ayunas > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resultado ECG en reposo (restecg)", [0, 1, 2])
    with col2:
        thalach = st.number_input("Frecuencia cardíaca máxima (thalach)", 50, 250, 150)
        exang = st.selectbox("Angina inducida por ejercicio (exang)", [0, 1])
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Pendiente ST (slope)", [1, 2, 3])
        ca = st.selectbox("Número de vasos principales (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (thal)", [3, 6, 7])

    submitted = st.form_submit_button("Realizar clasificación")

if submitted:
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    prediction = int(model.predict(input_data)[0])
    st.subheader("Resultado del modelo")

    if prediction == 0:
        st.success("Categoría estimada: 0 — sin presencia de enfermedad en la codificación del dataset.")
    else:
        st.info(
            f"Categoría estimada: {prediction} — clase positiva {prediction} "
            "en la codificación original del dataset."
        )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = model.named_steps["classifier"].classes_
        probability_df = pd.DataFrame({
            "Categoría": [str(int(c)) for c in classes],
            "Probabilidad estimada": probabilities
        }).set_index("Categoría")
        st.write("Probabilidades estimadas por categoría:")
        st.bar_chart(probability_df)

st.divider()
st.caption(
    "La salida reproduce las clases 0–4 de la variable target utilizada en el Notebook. "
    "Es una demostración académica y no una evaluación clínica."
)
