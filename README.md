# 🧬 Algoritmo Genético para Minería de Datos / Regresión

Proyecto que implementa un **algoritmo genético** para resolver problemas de **minería de datos**, como:
- Selección de características
- Ajuste de modelos de regresión lineal/no lineal
- Búsqueda de hiperparámetros  

Incluye scripts para preparar datos, ejecutar el algoritmo genético y evaluar los resultados.

---

## 📂 Contenido

- `README.md` — este archivo  
- `requirements.txt` — dependencias (si existe)  
- `data/` — carpeta con datasets de ejemplo (CSV)  
- `src/` — código fuente (algoritmo genético, evaluación, utilidades)  
- `notebooks/` — notebooks para experimentación y visualización  
- `scripts/` — scripts ejecutables (por ejemplo `run_experiment.py`, `train.py`)  
- `results/` — salida de ejecuciones (modelos, métricas, gráficos)  

> Ajusta la estructura anterior según los nombres reales de tus carpetas/archivos.

---

## ⚙️ Características

- Implementación **modular** de un Algoritmo Genético (AG)  
- Representación de **individuos (cromosomas)** para problemas de regresión y/o selección de atributos  
- Operadores genéticos: **selección, cruce (crossover), mutación y reemplazo**  
- Evaluación por métricas: `MSE`, `RMSE`, `R²`, etc.  
- Soporte para experimentar con distintos parámetros: tamaño de población, número de generaciones, tasa de mutación  

---

## 🧩 Requisitos

- Python **3.8+** (recomendado)
- Librerías principales: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`  

Ejemplo de `requirements.txt` mínimo:

```txt
numpy
pandas
scikit-learn
matplotlib
tqdm
