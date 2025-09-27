from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd

# Carpetas
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Columnas numéricas esperadas
NUMERIC_COLS = {
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
    "Tutoring_Sessions", "Physical_Activity", "Exam_Score",
}

# Binarias Yes/No
YES_NO_COLS = {"Extracurricular_Activities", "Internet_Access", "Learning_Disabilities"}
YES_NO_MAP = {"Yes": 1, "No": 0}

# Ordinales
ORDINAL_MAPS: Dict[str, Dict[str, int]] = {
    "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
    "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
    "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Family_Income": {"Low": 0, "Medium": 1, "High": 2},
    "Teacher_Quality": {"Low": 0, "Medium": 1, "High": 2},
    "Peer_Influence": {"Negative": 0, "Neutral": 1, "Positive": 2},
    "Parental_Education_Level": {"High School": 0, "College": 1, "Postgraduate": 2},
    "Distance_from_Home": {"Near": 0, "Moderate": 1, "Far": 2},
}

# Nominales (one-hot)
ONE_HOT_COLS = {"School_Type", "Gender"}

# Columnas a descartar si existieran
DROP_COLS = {"id"}

def cargar_csv(nombre: str) -> pd.DataFrame:
    ruta = RAW_DIR / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró data/raw/{nombre}")
    return pd.read_csv(ruta)

def aplicar_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    for c in YES_NO_COLS:
        if c in df.columns:
            df[c] = df[c].map(YES_NO_MAP).astype("Int64")
    return df

def aplicar_ordinales(df: pd.DataFrame) -> pd.DataFrame:
    for col, mapa in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapa).astype("Int64")
    return df

def aplicar_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ONE_HOT_COLS if c in df.columns]
    if cols:
        df = pd.get_dummies(df, columns=cols, drop_first=True)
    return df

def rellenar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            if df[c].isna().any():
                moda = df[c].mode(dropna=True)
                df[c] = df[c].fillna(moda.iloc[0] if not moda.empty else "Desconocido")
    return df

def asegurar_numericos(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    return df

# Pipeline principal

def preparar(nombre: str) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    df = cargar_csv(nombre).copy()

    # 0) quitar columnas irrelevantes si existen
    cols_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_drop:
        df = df.drop(columns=cols_drop)

    # 1) binarios Yes/No
    df = aplicar_yes_no(df)
    # 2) ordinales
    df = aplicar_ordinales(df)
    # 3) nominales -> one-hot
    df = aplicar_one_hot(df)
    # 4) nulos
    df = rellenar_nulos(df)
    # 5) asegurar numéricos
    df = asegurar_numericos(df)

    # 6) guardar
    salida = PROC_DIR / (Path(nombre).stem + "_clean.csv")
    df.to_csv(salida, index=False)

    resumen = {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "archivo_salida": str(salida),
        "columnas_eliminadas": cols_drop,
        "dummies_generadas": [c for c in df.columns if c.startswith("School_Type_") or c.startswith("Gender_")],
    }
    return df, resumen, str(salida)




































