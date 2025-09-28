# src/api/modelo.py
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = PROC_DIR / "model.pkl"

def _cargar_clean(nombre_clean: str) -> pd.DataFrame:
    """Lee el CSV ya limpio desde data/processed/."""
    ruta = PROC_DIR / nombre_clean
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró data/processed/{nombre_clean}")
    return pd.read_csv(ruta)

def _dividir_xy(df: pd.DataFrame, target: str = "Exam_Score") -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target}' en el dataset limpio.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def _rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))

def entrenar_y_guardar(
    nombre_clean: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Entrena dos modelos (Ridge y RandomForest), evalúa, elige el mejor por RMSE
    y guarda el mejor (model.pkl) junto con el orden de columnas.
    """
    df = _cargar_clean(nombre_clean)
    X, y = _dividir_xy(df, target="Exam_Score")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Modelos
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=random_state))
    ])

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state
    )

    # Entrenar
    ridge.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Evaluar
    preds_ridge = ridge.predict(X_test)
    preds_rf = rf.predict(X_test)

    metrics_ridge = {
        "MAE": float(mean_absolute_error(y_test, preds_ridge)),
        "RMSE": float(_rmse(y_test, preds_ridge)),
        "R2": float(r2_score(y_test, preds_ridge)),
        "modelo": "Ridge",
    }
    metrics_rf = {
        "MAE": float(mean_absolute_error(y_test, preds_rf)),
        "RMSE": float(_rmse(y_test, preds_rf)),
        "R2": float(r2_score(y_test, preds_rf)),
        "modelo": "RandomForest",
    }

    # Seleccionar mejor por RMSE
    mejor, obj = (metrics_rf, rf) if metrics_rf["RMSE"] < metrics_ridge["RMSE"] else (metrics_ridge, ridge)

    # Guardar modelo + metadatos mínimos
    payload = {
        "model": obj,
        "feature_names": list(X.columns),
        "target": "Exam_Score",
        "dataset": nombre_clean,
        "metrics": {"ridge": metrics_ridge, "random_forest": metrics_rf, "mejor": mejor},
    }
    joblib.dump(payload, model_path)

    return {
        "ok": True,
        "ruta_modelo": str(model_path),
        "dataset": nombre_clean,
        "metrics": {"ridge": metrics_ridge, "random_forest": metrics_rf, "mejor": mejor},
        "features": payload["feature_names"],
    }

def _cargar_modelo(model_path: Path = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}. Entrena primero.")
    return joblib.load(model_path)

def predecir(
    instancias: List[Dict[str, Any]],
    model_path: Path = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """
    Recibe una lista de instancias (dicts feature->valor) y devuelve predicciones.
    Reconciliamos columnas: faltantes -> 0; columnas extra -> se ignoran.
    """
    bundle = _cargar_modelo(model_path)
    model = bundle["model"]
    feature_names: List[str] = bundle["feature_names"]

    # Construir DataFrame desde el JSON
    X_in = pd.DataFrame(instancias)

    # Alinear columnas al orden esperado por el modelo
    for col in feature_names:
        if col not in X_in.columns:
            X_in[col] = 0 
    # Ignorar columnas no esperadas
    X_in = X_in[feature_names]

    preds = model.predict(X_in)
    return {"predicciones": [float(p) for p in preds], "n": len(preds)}
