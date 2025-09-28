# 📊 Proyecto Final — Análisis de Rendimiento Estudiantil

## 🎯 Problema
El objetivo de este proyecto es **predecir el puntaje de examen (`Exam_Score`) de estudiantes** a partir de factores académicos, socioeconómicos y personales.  
Se busca responder:  
- ¿Qué factores influyen más en el rendimiento estudiantil?  
- ¿Es posible construir un modelo que prediga con precisión los resultados de los exámenes?  

El proyecto integra:  
- **Exploración de datos** mediante una interfaz en Streamlit.  
- **API en FastAPI** para preparación de datos y modelado.  
- **Entrenamiento de modelos de predicción** (Regresión Ridge y Random Forest).  

---

## 📂 Datos
- **Fuente:** Dataset sintético de rendimiento estudiantil (`StudentPerformanceFactors.csv`).  
- **Origen (Kaggle):** https://www.kaggle.com/datasets/lainguyn123/student-performance-factors  
- **Ubicación:** `data/raw/`  
- **Procesamiento:**  
  - Codificación de variables categóricas (Yes/No, ordinales y nominales).  
  - Imputación de valores faltantes.  
  - Exportación de un dataset limpio en `data/processed/StudentPerformanceFactors_clean.csv`.

---

## ⚙️ Cómo correr el proyecto

### 1) Clonar el repositorio
```bash
git clone <URL_DEL_REPO>
cd proyecto_final_jsap
```

### 2) Crear y Activar el Entorno Virtual (Conda)
Es fundamental aislar las dependencias del proyecto. Usaremos Conda para crear el entorno base.

#### 1. Crear el entorno (lo nombraremos proyecto_ml):
```bash
conda create -n proyecto_ml python=3.11
```
#### 2. Activar el entorno:
```bash
conda activate proyecto_ml
```
#### 3) Instalar Dependencias (Poetry)
##### - Instalar Poetry (si no lo has hecho globalmente):
```bash
pip install poetry
```

##### - Instalar todas las dependencias del proyecto:
```bash
poetry install
```
## 📂 Estructura del proyecto

```plaintext
PROYECTO_FINAL_JSAP/
├── app/                        # Aplicación principal
│   └── app.py
│
├── data/                       # Datos del proyecto
│   ├── processed/              # Datos procesados y modelo entrenado
│   │   ├── model.pkl
│   │   └── StudentPerformanceFactors_clean.csv
│   └── raw/                    # Datos originales
│       └── StudentPerformanceFactors.csv
│
├── docs/                       # Documentación e imágenes
│   └── img/
│       ├── prediccion_modelo.png
│       ├── ui_inicial.png
│       └── ui_streamlit.png
│
├── notebooks/                  # Jupyter Notebooks
│   └── EDA.ipynb
│
├── src/                        # Código fuente del proyecto
│   └── api/                    # Módulo de la API
│       ├── main.py
│       ├── modelo.py
│       └── preparar_datos.py
│
├── tests/                      # Pruebas unitarias y contractuales
│   └── test_api_contract.py
│
├── .gitignore
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── pytest.ini
└── README.md
```
## Autores
- Juan Sebastian Angel Perez (juan.angel6@est.uexternado.edu.co)
- Diego Hernando Miranda Jimenez (diego.miranda1@est.uexternado.edu.co)

## 💻 Uso y Ejecución de la Aplicación
El sistema requiere que el servidor de la API y la interfaz de usuario se ejecuten simultáneamente en dos procesos separados.

**Paso 1: Iniciar el Servidor de la API**
En la Terminal 1 (Command Prompt), asegúrate de estar dentro del entorno de Poetry y navega al directorio que contiene main.py e inicia la API usando Uvicorn.
```bash
uvicorn src.api.main:app --reload --port 8000
```
**Resultado esperado:** El servidor debe mostrar un mensaje como: Uvicorn running on http://127.0.0.1:8000. Mantén esta terminal abierta y corriendo.

**Paso 2: Ejecutar la Interfaz de Streamlit (Frontend)**
Una vez que la API esté activa, abre la Terminal 2 (Command Prompt), activa el entorno de Poetry (si no lo está) y ejecuta la aplicación Streamlit:
```bash
python -m streamlit run app/app.py
```
Streamlit se abrirá automáticamente en tu navegador (normalmente en http://localhost:8501)

---
## 🌐 Endpoints de la API (FastAPI)
La API opera en http://127.0.0.1:8000. Los endpoints clave son:
| **Método** | **Endpoint** | **Descripción** | **Formato de Respuesta** |
| :---: | :--- | :--- | :--- |
| `GET` | `/health` | Verifica el estado del servidor. | `{"status": "ok"}` |
| `POST` | `/model/train` | Entrena el modelo usando los datos de entrada y lo guarda localmente. | JSON de confirmación |
| `POST` | `/model/predict` | **Predicción:** Recibe los datos de un estudiante y devuelve el puntaje estimado. | Valor numérico (o JSON con clave `predicciones`) |

---
## 📸 Capturas de pantalla
Agrega aquí tus capturas (colócalas en una carpeta `docs/img/` o `assets/` y enlázalas en markdown):

- [UI inicial](docs/img/ui_inicial.png)
- [Vista previa de los datos](docs/img/ui_streamlit.png)
- [Datos para las predicciones](docs/img/prediccion_modelo.png)  

## ⚙️ Interfaz de Usuario (Streamlit)
La aplicación app.py se divide en secciones para la exploración de datos y la interacción con el modelo.

**Interacción con el Modelo**
La **Sección 8: Predicción del Modelo** contiene un formulario completo con 20 campos de entrada. Es crucial que los valores ingresados coincidan con el mapeo que usa el modelo.

- Variables Numéricas: Ingreso directo (Ej: Horas de Estudio, Asistencia).
- Variables Ordinales (0, 1, 2): Se usan st.radio para asegurar que el valor numérico (0, 1 o 2) es enviado correctamente a la API, mientras se muestran etiquetas descriptivas al usuario (Ej: Baja, Media, Alta).

**Manejo de Errores de Conexión**
La aplicación ha sido diseñada para manejar fallos de conexión a la API.

Este error (WinError 10061) aparece si el servidor de la API no está corriendo o si la URL en streamlit_app.py es incorrecta. Asegúrate de ejecutar el Paso 1 antes de usar la aplicación.
