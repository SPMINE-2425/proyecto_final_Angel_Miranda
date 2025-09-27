from fastapi import FastAPI, HTTPException, Query
from src.api.preparar_datos import preparar

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

