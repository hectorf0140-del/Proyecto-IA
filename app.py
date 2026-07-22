import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="StayEdu", page_icon="🎓", layout="centered")

# Carga del modelo y los objetos guardados en el Paso 9
modelo = joblib.load("stayedu_model.pkl")
scaler = joblib.load("stayedu_scaler.pkl")
label_encoder = joblib.load("stayedu_label_encoder.pkl")
selected_features = joblib.load("stayedu_features.pkl")

# Etiquetas más claras para las variables más comunes de este dataset.
# Si una variable seleccionada no aparece aquí, se muestra su nombre original.
etiquetas = {
    "Curricular units 2nd sem (grade)": "Nota promedio 2do semestre",
    "Curricular units 1st sem (grade)": "Nota promedio 1er semestre",
    "Curricular units 2nd sem (approved)": "Materias aprobadas 2do semestre",
    "Curricular units 1st sem (approved)": "Materias aprobadas 1er semestre",
    "Curricular units 2nd sem (enrolled)": "Materias matriculadas 2do semestre",
    "Curricular units 1st sem (enrolled)": "Materias matriculadas 1er semestre",
    "Tuition fees up to date": "Matrícula al día (1 = Sí, 0 = No)",
    "Scholarship holder": "Tiene beca (1 = Sí, 0 = No)",
    "Debtor": "Tiene deudas pendientes (1 = Sí, 0 = No)",
    "Age at enrollment": "Edad al momento de inscribirse",
    "Admission grade": "Nota de admisión",
    "Gender": "Género (0 = Femenino, 1 = Masculino)",
    "Displaced": "Estudiante desplazado (1 = Sí, 0 = No)",
}

st.title("🎓 StayEdu")
st.write("Sistema de apoyo para estimar el riesgo de abandono académico.")

st.subheader("Datos del estudiante")

entrada = {}
with st.form("formulario_estudiante"):
    for var in selected_features:
        etiqueta = etiquetas.get(var, var)
        entrada[var] = st.number_input(etiqueta, value=0.0, step=1.0, format="%.2f")
    enviado = st.form_submit_button("Calcular riesgo")

if enviado:
    datos = pd.DataFrame([entrada])[selected_features]
    datos_escalados = scaler.transform(datos)

    prediccion = modelo.predict(datos_escalados)[0]
    probabilidades = modelo.predict_proba(datos_escalados)[0]

    clase_predicha = label_encoder.inverse_transform([prediccion])[0]

    # Se toma la probabilidad de la clase "Dropout" como referencia de riesgo
    if "Dropout" in label_encoder.classes_:
        idx_dropout = list(label_encoder.classes_).index("Dropout")
        prob_riesgo = probabilidades[idx_dropout]
    else:
        prob_riesgo = max(probabilidades)

    if prob_riesgo < 0.33:
        nivel = "Bajo"
        color = "green"
    elif prob_riesgo < 0.66:
        nivel = "Medio"
        color = "orange"
    else:
        nivel = "Alto"
        color = "red"

    st.subheader("Resultado")
    st.write(f"**Predicción del modelo:** {clase_predicha}")
    st.write(f"**Probabilidad de abandono:** {prob_riesgo * 100:.1f}%")
    st.markdown(f"**Nivel de riesgo:** :{color}[{nivel}]")

    st.subheader("Factores más relevantes (a nivel general del modelo)")
    if hasattr(modelo, "feature_importances_"):
        importancia = pd.Series(modelo.feature_importances_, index=selected_features)
        importancia = importancia.sort_values(ascending=False).head(5)
        for var, val in importancia.items():
            st.write(f"- {etiquetas.get(var, var)}")

    st.subheader("Recomendaciones generales")
    recomendaciones = {
        "Bajo": "El estudiante no muestra señales importantes de riesgo. Se recomienda dar seguimiento regular.",
        "Medio": "Se recomienda programar una tutoría de seguimiento y revisar su situación académica y económica.",
        "Alto": "Se recomienda contactar al estudiante lo antes posible y activar protocolos de apoyo académico y/o económico.",
    }
    st.info(recomendaciones[nivel])

    st.caption("Esta herramienta apoya la toma de decisiones y no reemplaza el criterio del personal académico.")
