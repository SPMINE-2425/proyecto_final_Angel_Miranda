from fastapi import FastAPI, HTTPException, Query
from src.api.preparar_datos import preparar
from typing import List, Dict, Any
from fastapi import Body
from src.api.modelo import entrenar_y_guardar, predecir

app = FastAPI(title="Proyecto Final - Seminario", version="0.1.0")

@app.get("/")
def raiz():
    return {"mensaje": "API en funcionamiento, bienvenido al Proyecto Final - Seminario"}

@app.get("/health")
def estado():
    return {"status": "ok"}

@app.get("/data/prepare")
def data_prepare(filename: str = Query(..., description="Nombre del CSV en data/raw")):
    """
    Prepara el CSV de data/raw y guarda *_clean.csv en data/processed.
    """
    try:
        _, resumen, _ = preparar(filename)
        return resumen
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al preparar: {e}")

# Endpoint para entrenar el modelo

@app.post("/model/train")
def model_train(
    filename: str = Query("StudentPerformanceFactors_clean.csv", description="Nombre del CSV limpio en data/processed")
):
    """
    Entrena Ridge y RandomForest con el dataset limpio, elige el mejor por RMSE,
    guarda el modelo en data/processed/model.pkl y devuelve métricas.
    """
    try:
        return entrenar_y_guardar(filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al entrenar: {e}")
    
# Endpoint para predecir con el modelo entrenado

@app.post("/model/predict")
def model_predict(
    payload: Dict[str, Any] = Body(..., description="JSON con 'instances': lista de objetos feature->valor")
):
    """
    Recibe:
    {
      "instances": [
        {"feature1": v1, "feature2": v2, ...},
        ...
      ]
    }
    Devuelve:
      {"predicciones": [..], "n": N}
    """
    try:
        if "instances" not in payload or not isinstance(payload["instances"], list):
            raise ValueError("El cuerpo debe incluir 'instances' como lista de objetos.")

        return predecir(payload["instances"])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")