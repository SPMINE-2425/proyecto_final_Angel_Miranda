import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from src.api.main import app

client = TestClient(app)


# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def ensure_model_trained():
    """
    Garantiza que exista un modelo entrenado antes de probar /model/predict.
    - Intenta entrenar usando el limpio por defecto: StudentPerformanceFactors_clean.csv
    - Si no existe el limpio, omite las pruebas de predicción con un mensaje claro.
    """
    # Entrena el modelo
    resp = client.post("/model/train", params={"filename": "StudentPerformanceFactors_clean.csv"})
    if resp.status_code == 404:
        pytest.skip(
            "No se encontró data/processed/StudentPerformanceFactors_clean.csv. "
            "Primero corre GET /data/prepare con filename=StudentPerformanceFactors.csv."
        )
    assert resp.status_code == 200, f"Fallo al entrenar el modelo: {resp.text}"
    data = resp.json()
    assert "ruta_modelo" in data and "metrics" in data, "Respuesta de /model/train no tiene campos esperados."
    return data


# ---------- Pruebas mínimas ----------

def test_health_ok():
    """Validar que /health responde 200 y el payload básico."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    # Contrato mínimo esperado
    assert isinstance(body, dict)
    assert body.get("status") == "ok"


def test_predict_contract_schema(ensure_model_trained):
    """
    Validar el contrato de /model/predict:
    - Responde 200
    - Devuelve un dict con llaves 'predicciones' (list) y 'n' (int)
    Nota: se envía una instancia vacía; el backend completa columnas faltantes con 0.
    """
    payload = {"instances": [{}]}  # Smoke payload genérico
    resp = client.post("/model/predict", json=payload)
    assert resp.status_code == 200, f"Respuesta inesperada: {resp.text}"

    body = resp.json()
    assert isinstance(body, dict), "El cuerpo debe ser un dict"
    assert "predicciones" in body and "n" in body, "Faltan llaves en la respuesta"
    assert isinstance(body["predicciones"], list), "'predicciones' debe ser una lista"
    assert isinstance(body["n"], int), "'n' debe ser entero"
    # Si hay al menos una instancia, debe haber al menos una predicción numérica
    if payload["instances"]:
        assert len(body["predicciones"]) == body["n"] >= 1
        assert all(isinstance(x, (int, float)) for x in body["predicciones"])


def test_predict_smoke_example(ensure_model_trained):
    """
    Smoke test: corre /model/predict con un caso de ejemplo simple.
    Se usa un payload con algunos campos típicos; el backend completará el resto con 0.
    """
    example = {
        "Hours_Studied": 10,
        "Attendance": 92,
        "Sleep_Hours": 7,
        "Previous_Scores": 75,
        "Tutoring_Sessions": 1,
        "Physical_Activity": 3,
        # Binarias (1/0)
        "Extracurricular_Activities": 1,
        "Internet_Access": 1,
        "Learning_Disabilities": 0,
        # Ordinales ya codificados (0/1/2)
        "Parental_Involvement": 2,
        "Access_to_Resources": 1,
        "Motivation_Level": 2,
        "Family_Income": 1,
        "Teacher_Quality": 2,
        "Peer_Influence": 1,
        "Parental_Education_Level": 1,
        "Distance_from_Home": 1,
        # Dummies nominales (ejemplos; si no existen en el modelo, se ignoran)
        "School_Type_Public": 1,
        "Gender_Male": 0,
    }

    resp = client.post("/model/predict", json={"instances": [example]})
    assert resp.status_code == 200, f"Respuesta inesperada: {resp.text}"

    body = resp.json()
    assert isinstance(body, dict)
    assert "predicciones" in body and "n" in body
    assert body["n"] == 1
    assert isinstance(body["predicciones"], list) and len(body["predicciones"]) == 1
    assert isinstance(body["predicciones"][0], (int, float))