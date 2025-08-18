# imputacion_iterative_airquality.py
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# scikit-learn imports
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# ---------------------------
# Configuración
# ---------------------------
data_path = Path(r"C:\Users\JJ\Documents\Mineria_Datos_AlgGenetico\Datasets\AirQuality.csv")
drop_threshold = 0.5   # Eliminar columnas con >50% NaN
target_col = "CO(GT)"  # Variable objetivo (se eliminarán filas con target faltante)
max_iter = 10
random_state = 0
# ---------------------------

if not data_path.exists():
    raise FileNotFoundError(f"No se encuentra el archivo: {data_path}")

# 1. Lectura del CSV con separador y decimal correctos
df = pd.read_csv(
    data_path,
    sep=';',          # separador de columnas
    decimal=',',      # coma como separador decimal
    engine='python'
)

# 2. Eliminar columnas completamente vacías
df = df.dropna(axis=1, how='all')

# 2. Eliminar columnas de fecha/hora si existen
for col in ["Date", "Time"]:
    if col in df.columns:
        df = df.drop(columns=[col])


# 3. Convertir cadenas con coma decimal a numéricas cuando sea posible
for col in df.columns:
    if df[col].dtype == object:
        cleaned = df[col].astype(str).str.replace(',', '.').str.strip()
        coerced = pd.to_numeric(cleaned, errors='coerce')
        if coerced.notna().sum() > 0:
            df[col] = coerced

# 4. Reemplazar valores -200 (faltantes en dataset original) por NaN
df = df.replace(-200, np.nan)

# 5. Detectar columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

print("Columnas numéricas detectadas:", numeric_cols)
print("Columnas no numéricas detectadas (se excluyen de imputación):", non_numeric_cols)

# 6. Eliminar filas con target faltante (si se usará para modelado)
if target_col in df.columns:
    n_before = len(df)
    df = df.dropna(subset=[target_col])
    n_after = len(df)
    print(f"Eliminadas {n_before - n_after} filas con '{target_col}' faltante.")

# 7. Eliminar columnas con más de drop_threshold de NaN
missing_frac = df.isna().mean()
cols_to_drop = missing_frac[missing_frac > drop_threshold].index.tolist()
if cols_to_drop:
    print(f"Eliminando columnas con >{int(drop_threshold*100)}% faltantes: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    numeric_cols = [c for c in numeric_cols if c not in cols_to_drop]

# 8. Definir columnas a imputar (numéricas excepto target)
impute_cols = [c for c in numeric_cols if c != target_col]
print("Columnas que serán imputadas:", impute_cols)

# 9. Imputación
if len(impute_cols) == 0:
    warnings.warn("No hay columnas numéricas para imputar.")
else:
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        initial_strategy='median',
        random_state=random_state,
        sample_posterior=False
    )

    imputed_array = imputer.fit_transform(df[impute_cols])
    df_imputed = df.copy()
    df_imputed[impute_cols] = imputed_array

    # 10. Guardar el resultado
    out_path = data_path.with_name(data_path.stem + "_imputed_iterative.csv")
    df_imputed.to_csv(out_path, index=False)
    print(f"✅ Imputación completada. Archivo guardado en: {out_path}")

    # 11. Comprobar valores faltantes
    print("Valores faltantes después de imputar (debe ser 0):")
    print(df_imputed[impute_cols].isna().sum())
