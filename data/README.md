# Carpeta `data/`

Este directorio contiene los conjuntos de datos utilizados en el proyecto.  
La división en subcarpetas ayuda a mantener un flujo de trabajo ordenado y reproducible.

## 📂 Estructura

- `raw/`  
  Aquí se almacenan los **datos originales** tal como se obtuvieron de la fuente.  
  - No deben modificarse manualmente.  
  - Se incluyen aquí en su formato bruto (CSV, Excel, JSON, etc.).  

- `processed/`  
  Contiene los **datos limpios, transformados y particionados** que se generen durante el procesamiento.  
  - Por ejemplo: normalizaciones, filtros, muestreos, particiones *train/test*.  
  - Estos datos son los que se utilizan directamente en los modelos o análisis.

## 📝 Buenas prácticas

- Nunca modificar los archivos en `raw/`. Si se necesita limpieza o transformación, crear una copia en `processed/`.  
- Documentar cualquier cambio realizado en los datos (ej. normalización, eliminación de nulos, creación de variables).  
- Mantener un tamaño razonable de archivos para que el repositorio siga siendo manejable.  
  - Si los datos son muy grandes, considerar almacenar solo una muestra representativa y documentar la ubicación del dataset completo.

## 🚨 Notas

- Verificar siempre las **licencias y permisos** antes de subir datos al repositorio.  
- En caso de datos confidenciales o privados, no subirlos aquí; solo incluir scripts que expliquen cómo obtenerlos o procesarlos.
