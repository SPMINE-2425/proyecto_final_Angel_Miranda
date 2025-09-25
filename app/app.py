import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

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
        st.subheader("5) Histograma")
        col_sel = st.selectbox("Columna numérica", num_cols)
        bins = st.slider("Número de bins", 5, 100, 30)
        st.bar_chart(pd.cut(df[col_sel], bins=bins).value_counts().sort_index())

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

else:
    st.info("Selecciona un archivo de la lista o sube un CSV para continuar.")
