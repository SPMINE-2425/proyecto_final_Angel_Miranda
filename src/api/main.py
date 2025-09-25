from fastapi import FastAPI

app = FastAPI(title="Proyecto Final - Seminario", version="0.1.0")

@app.get("/")
def raiz():
    return {"mensaje": "API en funcionamiento, bienvenido al Proyecto Final - Seminario"}

@app.get("/health")
def estado():
    return {"status": "ok"}