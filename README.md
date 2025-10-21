Algoritmo Genético para Minería de Datos / Regresión
Proyecto que implementa un algoritmo genético para resolver problemas de minería de datos (por ejemplo: selección de características, ajuste de modelos de regresión lineal/no lineal y búsqueda de hiperparámetros). Contiene scripts para preparar datos, ejecutar el algoritmo genético y evaluar sus resultados.

Contenido

README.md — este archivo.

requirements.txt — dependencias (si existe).

data/ — carpeta con datasets de ejemplo (CSV).

src/ — código fuente (algoritmo genético, evaluación, utilidades).

notebooks/ — notebooks para experimentación y visualización.

scripts/ — scripts ejecutables (por ejemplo run_experiment.py, train.py).

results/ — salida de ejecuciones (modelos, métricas, gráficos).

Ajusta la estructura anterior según los nombres reales de tus carpetas/archivos.

Características

Implementación básica/modular de un Algoritmo Genético (AG).

Representación de individuos (cromosomas) para problemas de regresión y/o selección de atributos.

Operadores genéticos: selección, cruce (crossover), mutación y reemplazo.

Evaluación por métricas (MSE, RMSE, R², etc.).

Soporte para experimentar con distintos parámetros (tamaño de población, número de generaciones, tasa de mutación).

Requisitos

Python 3.8+ (recomendado)

Paquetes habituales: numpy, pandas, scikit-learn, matplotlib (añade otros si tu repo los usa)

Ejemplo de requirements.txt mínimo:

numpy
pandas
scikit-learn
matplotlib
tqdm

Instalación

Clona el repositorio:

git clone https://github.com/topclutch/Alggenreglin.git
cd Alggenreglin/Mineria_Datos_AlgGenetico


Crea un entorno virtual e instala dependencias:

python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
pip install -r requirements.txt


Si no tienes requirements.txt, instala las librerías necesarias manualmente:

pip install numpy pandas scikit-learn matplotlib tqdm

Uso
1) Preparar datos

Coloca tus datasets en data/. Se asume formato CSV con la columna objetivo (por ejemplo target) y las columnas de características.

Ejemplo de preprocesamiento (si tienes un script prepare_data.py):

python src/prepare_data.py --input data/raw/dataset.csv --output data/processed/dataset_proc.csv

2) Ejecutar algoritmo genético

Ejemplo genérico de ejecución (ajusta el nombre del script y parámetros según tu repo):

python src/run_genetic.py \
  --dataset data/processed/dataset_proc.csv \
  --target target \
  --pop_size 100 \
  --generations 200 \
  --crossover_rate 0.8 \
  --mutation_rate 0.01 \
  --output results/run1


Parámetros comunes:

--dataset: ruta al CSV.

--target: nombre de la columna objetivo.

--pop_size: tamaño de la población.

--generations: número de generaciones.

--crossover_rate: probabilidad de cruce.

--mutation_rate: probabilidad de mutación.

--seed: semilla aleatoria (reproducibilidad).

--output: carpeta donde guardar resultados y modelos.

3) Evaluación y visualización

Si existe un script de evaluación:

python src/evaluate.py --results results/run1 --metric rmse


O abre los notebooks en notebooks/ para visualizar convergencia, histogramas, comparaciones, etc.

Ejemplos (Salida esperada)

Carpeta results/run1/ con:

model.pkl (modelo guardado)

metrics.json (métricas de desempeño)

history.csv (fitness por generación)

gráficos de convergencia (convergence.png)

Sugerencias para experimentar

Prueba distintos tamaños de población y número de generaciones para ver efecto en convergencia.

Ajusta la tasa de mutación: valores pequeños (0.001–0.01) suelen mantener diversidad sin desordenar las soluciones.

Intenta distintas funciones de fitness (por ejemplo, MSE penalizado por número de variables para favorecer soluciones más simples).

Usa validación cruzada (k-fold) al evaluar fitness para evitar sobreajuste.

Estructura del código (sugerida)

Describe brevemente las partes principales del código. Ejemplo:

src/
  genetic/               # implementación del algoritmo genético
    population.py
    individual.py
    operators.py
    fitness.py
  data/                  # carga y preprocesamiento
    loader.py
    preprocess.py
  experiments/           # scripts para correr experimentos
    run_genetic.py
  utils/
    io.py
    plotting.py

Contribuciones

Haz fork del repositorio.

Crea una rama para tu feature: git checkout -b feature/mi-cambio.

Haz commits claros y push.

Abre un Pull Request describiendo tus cambios.

Licencia

Indica la licencia usada (por ejemplo MIT). Si todavía no hay licencia, añade una, por ejemplo:

MIT License


Email: tu.email@dominio.com (opcional)

Issues: abre un issue en el repositorio para preguntas o bugs.
