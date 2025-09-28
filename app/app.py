import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import requests
import altair as alt

API_URL_PREDICCION = "http://127.0.0.1:8000/model/predict"

# Metadatos y título
st.set_page_config(page_title="Proyecto Final — Análisis de Rendimiento Estudiantil", page_icon="📊")
st.title("Proyecto Final — Análisis de Rendimiento Estudiantil")
st.caption("Carga un CSV desde data/raw o súbelo y visualiza un resumen rápido.")

# Carpeta de datos
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_DIR.exists():
    st.error("La carpeta data/raw no existe. Verifica la estructura del proyecto.")
    st.stop()

# Función para llamar a la API de predicción

def obtener_prediccion(data_payload: dict):
    """Realiza la llamada POST a la API de predicción."""
    try:
        # Realizar la solicitud POST
        response = requests.post(
            API_URL_PREDICCION, 
            json=data_payload, 
            timeout=10 # Límite de tiempo para la respuesta
        )
        
        # Verificar si la solicitud fue exitosa (código 200)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error en la API: Código de estado {response.status_code}")
            st.json(response.json()) # Muestra el mensaje de error de la API
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión con la API ({API_URL_PREDICCION}): {e}")
        st.warning("Asegúrate de que tu servicio de API esté corriendo y la URL sea correcta.")
        return None

# Selección / carga de archivo
st.subheader("1) Selecciona o sube un archivo CSV")

# Elegir un archivo ya existente en data/raw
archivos = sorted([p.name for p in RAW_DIR.glob("*.csv")])
opcion = st.selectbox("Archivos en data/raw", ["(ninguno)"] + archivos, index=0)

# Subir un archivo nuevo
subido = st.file_uploader("O subir un CSV", type=["csv"])

# Cargar DataFrame
df = None
fuente = None

if subido is not None:
    # Lee el CSV subido
    try:
        df = pd.read_csv(subido)
        fuente = f"archivo subido: {subido.name}"
        st.success(f"CSV cargado correctamente desde el cargador ({subido.name}).")
    except Exception as e:
        st.error(f"No se pudo leer el archivo subido: {e}")

elif opcion != "(ninguno)":
    # Lee el CSV seleccionado de data/raw
    try:
        df = pd.read_csv(RAW_DIR / opcion)
        fuente = f"archivo en data/raw: {opcion}"
        st.success(f"CSV cargado correctamente: {opcion}")
    except Exception as e:
        st.error(f"No se pudo leer '{opcion}': {e}")

# Guardar archivo subido en data/processed
if subido is not None and df is not None:
    guardar = st.checkbox("Guardar archivo subido en data/processed", value=True)
    if guardar and st.button("Guardar"):
        destino = PROC_DIR / subido.name
        # Evita sobrescribir: si ya existe, agrega timestamp
        if destino.exists():
            destino = destino.with_name(f"{destino.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(destino, "wb") as f:
            f.write(subido.getvalue())
        st.success(f"Archivo guardado como: {destino.name}")


# Resumen y vista previa
if df is not None:
    st.markdown(f"**Fuente:** {fuente}")

    st.subheader("2) Resumen del dataset")
    resumen = {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "nombres_columnas": list(df.columns),
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
    }
    st.json(resumen)

    st.subheader("3) Vista previa (primeras 10 filas)")
    st.dataframe(df.head(10), use_container_width=True)

    # Estadísticas específicas Exam_Score
    if "Exam_Score" in df.columns:
        st.subheader("4) Estadísticas de 'Exam_Score'")
        st.write(df["Exam_Score"].describe())

    # Histograma
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    if num_cols:
        st.subheader("5) Histograma de Frecuencias")
        
        # Widgets para selección
        col_hist_selector, col_bin_slider = st.columns(2)
        with col_hist_selector:
            col_sel = st.selectbox("Columna numérica", num_cols, key="hist_col")
        with col_bin_slider:
            bins = st.slider("Número de bins", 5, 100, 30, key="hist_bins")
       
        try:
            data_counts = pd.cut(df[col_sel], bins=bins).value_counts().sort_index()

            df_chart = data_counts.rename_axis('Rango_Valores').reset_index(name='Conteo')
            
            df_chart['Rango_Valores'] = df_chart['Rango_Valores'].astype(str)

            st.bar_chart(df_chart, x='Rango_Valores', y='Conteo')

        except Exception as e:
            st.warning(f"No se pudo generar el Histograma para '{col_sel}'. Error: {e}")
            st.info("Verifica que la columna seleccionada no contenga valores nulos o atípicos que dificulten la división en bins.")

    # Dispersión
    if len(num_cols) >= 2:
        st.subheader("6) Dispersión (scatter)")
        c1 = st.selectbox("Eje X", num_cols, key="xcol")
        c2 = st.selectbox("Eje Y", num_cols, key="ycol")
        st.caption("Muestra hasta 2,000 puntos para mantener fluidez.")
        sample = df[[c1, c2]].dropna().sample(min(2000, len(df)), random_state=42)
        st.scatter_chart(sample.rename(columns={c1: "x", c2: "y"}))

    # Correlación
    if len(num_cols) >= 2:
        st.subheader("7) Correlación (Pearson)")
        st.dataframe(df[num_cols].corr().round(3), use_container_width=True)

    # Predicción con API
    st.markdown("---")
    st.header("8) Predicción del Modelo")
    st.info("Ingresa los parámetros necesarios y pulsa el botón para enviar los datos a la API de predicción.")

    # 1. Crear el formulario de entrada de datos
    with st.form("prediccion_form"):
        st.markdown("### Parámetros de Entrada del Estudiante")
        
        # --- FILA 1 (Numéricas) ---
        col1, col2, col3 = st.columns(3)
        with col1:
            hours_studied = st.number_input("Horas de Estudio (Hours_Studied)", min_value=0, max_value=50, value=5)
        with col2:
            attendance = st.slider("Asistencia (%) (Attendance)", min_value=0, max_value=100, value=90)
        with col3:
            sleep_hours = st.number_input("Horas de Sueño (Sleep_Hours)", min_value=1, max_value=12, value=7)
        
        # --- FILA 2 (Numéricas) ---
        col4, col5, col6 = st.columns(3)
        with col4:
            previous_scores = st.number_input("Puntajes Previos (Previous_Scores)", min_value=0, max_value=100, value=70)
        with col5:
            tutoring_sessions = st.number_input("Sesiones Tutoría (Tutoring_Sessions)", min_value=0, max_value=10, value=2)
        with col6:
            physical_activity = st.number_input("Actividad Física (Physical_Activity)", min_value=0, max_value=7, value=3)

        # --- FILA 3 (Binarias) ---
        col7, col8, col9 = st.columns(3)
        with col7:
            # YES_NO_MAP = {"Yes": 1, "No": 0}
            extracurricular_activities = st.radio("Actividades Extracurriculares (Extracurricular_Activities)", options=[1, 0], index=0, format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
        with col8:
            # YES_NO_MAP = {"Yes": 1, "No": 0}
            internet_access = st.radio("Acceso a Internet (Internet_Access)", options=[1, 0], index=0, format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
        with col9:
            # YES_NO_MAP = {"Yes": 1, "No": 0}
            learning_disabilities = st.radio("Discapacidades de Aprendizaje (Learning_Disabilities)", options=[0, 1], index=0, format_func=lambda x: "No (0)" if x == 0 else "Sí (1)")

        st.markdown("#### Variables Ordinales (Escala 0-2)")
        # --- FILA 4 (Ordinales 0, 1, 2) ---
        col10, col11, col12 = st.columns(3)
        
        MAP_LOW_MED_HIGH = {0: "Baja (0)", 1: "Media (1)", 2: "Alta (2)"}
        MAP_LOW_MED_HIGH_HELP = "Mapeo: Baja=0, Media=1, Alta=2"

        with col10:
            parental_involvement = st.radio("Participación Parental (Parental_Involvement)", 
                                           options=[0, 1, 2], index=1, 
                                           format_func=lambda x: MAP_LOW_MED_HIGH[x],
                                           help=MAP_LOW_MED_HIGH_HELP)
        with col11:
            access_to_resources = st.radio("Acceso a Recursos (Access_to_Resources)", 
                                         options=[0, 1, 2], index=1, 
                                         format_func=lambda x: MAP_LOW_MED_HIGH[x],
                                         help=MAP_LOW_MED_HIGH_HELP)
        with col12:
            motivation_level = st.radio("Nivel de Motivación (Motivation_Level)", 
                                      options=[0, 1, 2], index=1, 
                                      format_func=lambda x: MAP_LOW_MED_HIGH[x],
                                      help=MAP_LOW_MED_HIGH_HELP)

        # --- FILA 5 (Ordinales 0, 1, 2) ---
        col13, col14, col15 = st.columns(3)
        
        with col13:
            # Family_Income (Low: 0, Medium: 1, High: 2)
            family_income = st.radio("Ingreso Familiar (Family_Income)", 
                                     options=[0, 1, 2], index=1, 
                                     format_func=lambda x: MAP_LOW_MED_HIGH[x],
                                     help=MAP_LOW_MED_HIGH_HELP)
        with col14:
            # Teacher_Quality (Low: 0, Medium: 1, High: 2)
            teacher_quality = st.radio("Calidad del Profesor (Teacher_Quality)", 
                                       options=[0, 1, 2], index=1, 
                                       format_func=lambda x: MAP_LOW_MED_HIGH[x],
                                       help=MAP_LOW_MED_HIGH_HELP)
        with col15:
            # Peer_Influence (Negative: 0, Neutral: 1, Positive: 2)
            MAP_PEER = {0: "Negativa (0)", 1: "Neutra (1)", 2: "Positiva (2)"}
            peer_influence = st.radio("Influencia de Pares (Peer_Influence)", 
                                      options=[0, 1, 2], index=1, 
                                      format_func=lambda x: MAP_PEER[x],
                                      help="Mapeo: Negativa=0, Neutra=1, Positiva=2")

        # --- FILA 6 (Ordinales y One-Hot) ---
        col16, col17, col18 = st.columns(3)
        
        with col16:
            # Parental_Education_Level (High School: 0, College: 1, Postgraduate: 2)
            MAP_EDUCATION = {0: "Bachillerato (0)", 1: "Universidad (1)", 2: "Posgrado (2)"}
            parental_education_level = st.radio("Nivel Educación Parental (Parental_Education_Level)", 
                                                options=[0, 1, 2], index=1, 
                                                format_func=lambda x: MAP_EDUCATION[x],
                                                help="Mapeo: Bachillerato=0, Universidad=1, Posgrado=2")
        with col17:
            # Distance_from_Home (Near: 0, Moderate: 1, Far: 2)
            MAP_DISTANCE = {0: "Corta (0)", 1: "Moderada (1)", 2: "Larga (2)"}
            distance_from_home = st.radio("Distancia a Casa (Distance_from_Home)", 
                                          options=[0, 1, 2], index=1, 
                                          format_func=lambda x: MAP_DISTANCE[x],
                                          help="Mapeo: Corta=0, Moderada=1, Larga=2")
        with col18:
            # School_Type (Public: 1, Private: 0)
            school_type_public = st.radio("Tipo Escuela (School_Type_Public)", 
                                          options=[1, 0], index=0, 
                                          format_func=lambda x: "Pública (1)" if x == 1 else "Privada (0)")
        
        # Gender (Male: 1, Female: 0)
        gender_male = st.radio("Género (Gender_Male)", 
                               options=[0, 1], index=0, 
                               format_func=lambda x: "Femenino (0)" if x == 0 else "Masculino (1)")


        # Botón para enviar la solicitud
        submitted = st.form_submit_button("Obtener Predicción de Rendimiento")

        if submitted:
            # 2. Recolectar datos en el formato JSON esperado por la API (el formato 'instances' es crucial)
            datos_instancia = {
                "Hours_Studied": hours_studied,
                "Attendance": attendance,
                "Sleep_Hours": sleep_hours,
                "Previous_Scores": previous_scores,
                "Tutoring_Sessions": tutoring_sessions,
                "Physical_Activity": physical_activity,
                "Extracurricular_Activities": extracurricular_activities,
                "Internet_Access": internet_access,
                "Learning_Disabilities": learning_disabilities,
                "Parental_Involvement": parental_involvement,
                "Access_to_Resources": access_to_resources,
                "Motivation_Level": motivation_level,
                "Family_Income": family_income,
                "Teacher_Quality": teacher_quality,
                "Peer_Influence": peer_influence,
                "Parental_Education_Level": parental_education_level,
                "Distance_from_Home": distance_from_home,
                "School_Type_Public": school_type_public,
                "Gender_Male": gender_male,
            }
            
            # Estructura final con la clave 'instances'
            datos_a_enviar = {
                "instances": [datos_instancia]
            }

            with st.spinner("Enviando datos y esperando predicción..."):
                # 3. Llamar a la función que interactúa con la API
                resultado_api = obtener_prediccion(datos_a_enviar)

            # 4. Mostrar el resultado
            if resultado_api:
                st.success("✅ Predicción recibida con éxito:")
                
                # Manejamos la respuesta, asumiendo que la predicción viene en el primer elemento de la lista
                if 'predicciones' in resultado_api and isinstance(resultado_api['predicciones'], list):
                    # Asumiendo que el modelo devuelve una lista de predicciones, tomamos la primera
                    prediccion = resultado_api['predicciones'][0]
                    st.metric(
                        label="Puntaje de Examen Estimado (Exam_Score)", 
                        value=f"{prediccion:.2f}",
                        help="El valor predicho para el rendimiento estudiantil."
                    )
                else:
                    # Caso de fallback si el JSON de respuesta tiene un formato inesperado
                    st.warning("La API respondió, pero el formato de la predicción fue inesperado (buscaba la clave 'predicciones').")
                
                # Muestra la respuesta completa de la API para debug
                with st.expander("Ver respuesta completa de la API (Debugging)"):
                    st.json(resultado_api)
    
else:
    st.info("Selecciona un archivo de la lista o sube un CSV para continuar.")
