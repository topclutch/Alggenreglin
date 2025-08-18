# Librerias
import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# ------------------ Función que evalúa UN individuo (vector binario) ------------------
def evaluar_individuo(bin_vector, X_df, y_series, test_size=0.2, random_state=42):
    """
    Recibe un vector binario (list o array) que indica qué columnas de X_df usar.
    Devuelve (r2, adjusted_r2).
    Si no hay variables seleccionadas devuelve r2=0 y adjusted_r2 = -inf (evitar selección).
    """
    indices = [i for i, b in enumerate(bin_vector) if int(b) == 1]
    p = len(indices)
    # Si no hay predictores seleccionados, no se puede ajustar modelo
    if p == 0:
        return 0.0, -np.inf

    X_sub = X_df.iloc[:, indices]

    # split
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_series, test_size=test_size, random_state=random_state)
    # ajustar
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # r2 en el conjunto de test
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    
    # Reestriccion
    if (r2) < 0.6:
        adjusted_r2 = 0
    else:
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    return r2,adjusted_r2

# ------------------ Inicializar población y calcular aptitud (AHORA correctamente) ------------------
def poblacion_inicial_y_aptitud(X_df, y_series, p_original, numero_individuos, test_size=0.2, random_state=42):
    if numero_individuos % 2 != 0:
        raise ValueError("Error: El número de individuos debe ser par.")
    num_variables = p_original
    # generar población aleatoria
    poblacion = np.random.randint(0, 2, size=(numero_individuos, num_variables))
    individuos_info = []

    print("1.- Inicializar la Población y cálculo de Aptitud (Adjusted R2)\n")
    for idx, individuo in enumerate(poblacion):
        r2_ind, adj_r2_ind = evaluar_individuo(individuo, X_df, y_series, test_size=test_size, random_state=random_state)
        
        individuos_info.append({
            "Individuo": idx + 1,
            "Vector binario": list(individuo),
            "Peso_suma": r2_ind,      # R2 del subconjunto
            "Ganancia": adj_r2_ind   # Adjusted R2 (aptitud)
        })

    # imprimir resumen
    print("\n" + "-" * 70)
    print(f"{'Individuo':>10} | {'Vector binario':<25} | {'R2':>10} | {'R2 Adjusted':>17}")
    print("-" * 70)
    for info in individuos_info:
        vec_str = ''.join(str(bit) for bit in info["Vector binario"])
        # formatear adjusted R2 si es -inf
        adj = info["Ganancia"]
        adj_str = f"{adj:.6f}" if np.isfinite(adj) else str(adj)
        print(f"{info['Individuo']:>10} | {vec_str:<25} | {info['Peso_suma']:>10.6f} | {adj_str:>17}")
    print("-" * 70)
    return individuos_info

# ------------------ Algoritmo genético (uso evaluar_individuo para los hijos) ------------------
def ejecutar_algoritmo_genetico(individuos_info, X_df, y_series, n_observations, num_individuos, pc, pm, n=30, test_size=0.2, random_state=42):
    mejores_soluciones = []
    for generacion in range(n):
        print(f"\n========= Generación {generacion + 1} =========")
        # Selección por torneo
        print("2.- Selección de Padres (Tournament)\n")
        padres = []
        while len(padres) < num_individuos:
            a, b = random.sample(individuos_info, 2)
            # elegir mayor aptitud (Ganancia)
            if a['Ganancia'] > b['Ganancia']:
                padres.append(a)
            elif b['Ganancia'] > a['Ganancia']:
                padres.append(b)
            else:
                padres.append(random.choice([a, b]))

        # Cruce por pares
        print("3.- Cruzamiento por 2 puntos(bits)\n")
        hijos = []
        for i in range(0, len(padres), 2):
            padre1 = padres[i]["Vector binario"].copy()
            padre2 = padres[i+1]["Vector binario"].copy()
            r = random.random()
            if r <= pc:
                puntos_corte = [j + 0.5 for j in range(len(padre1) - 1)]
                p1, p2 = random.sample(puntos_corte, 2)
                idx1 = int(p1 + 0.5)
                idx2 = int(p2 + 0.5)
                if p1 < p2:
                    centro1 = padre1[idx1:idx2]
                    centro2 = padre2[idx1:idx2]
                    hijo1 = padre1[:idx1] + centro2 + padre1[idx2:]
                    hijo2 = padre2[:idx1] + centro1 + padre2[idx2:]
                else:
                    # caso simétrico (aunque raro con esta selección)
                    extremo1_izq = padre1[:idx2]
                    extremo1_der = padre1[idx1:]
                    extremo2_izq = padre2[:idx2]
                    extremo2_der = padre2[idx1:]
                    # las rebanadas intermedias pueden ser vacías
                    hijo1 = extremo2_izq + padre1[idx2:idx1] + extremo2_der
                    hijo2 = extremo1_izq + padre2[idx2:idx1] + extremo1_der
            else:
                hijo1 = padre1
                hijo2 = padre2
            hijos.append(hijo1)
            hijos.append(hijo2)

        # Mutación
        print("4.- Mutacion \n")
        for i in range(len(hijos)):
            nuevo_hijo = []
            for bit in hijos[i]:
                if random.random() <= pm:
                    nuevo_hijo.append(1 if bit == 0 else 0)
                else:
                    nuevo_hijo.append(bit)
            hijos[i] = nuevo_hijo

        # Evaluar hijos (ahora con evaluación correcta por subconjunto)
        print("5.- Evaluación de Hijos \n")
        hijos_info = []
        for idx, hijo in enumerate(hijos):
            r2_h, adj_r2_h = evaluar_individuo(hijo, X_df, y_series, test_size=test_size, random_state=random_state)
            hijos_info.append({
                "Individuo": idx + 1,
                "Vector binario": list(hijo),
                "Peso_suma": r2_h,
                "Ganancia": adj_r2_h
            })

        # Mostrar mejor hijo de la generación
        mejor_hijo = max(hijos_info, key=lambda x: x["Ganancia"])
        mejores_soluciones.append(mejor_hijo)
        adj_str = f"{mejor_hijo['Ganancia']:.6f}" if np.isfinite(mejor_hijo['Ganancia']) else str(mejor_hijo['Ganancia'])
        print(f"\n>>> Mejor individuo de esta generación: Hijo {mejor_hijo['Individuo']} con Adjusted R2 = {adj_str}")
        print(f"Vector binario: {mejor_hijo['Vector binario']}\n")

        # Reemplazo: la nueva población serán los hijos evaluados
        individuos_info = hijos_info

    return mejores_soluciones

# ------------------función para regresión con todas las variables (diagnóstico) ------------------
def regresion_lineal_multiple(df, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Coeficientes:", modelo.coef_)
    print("Intercepto:", modelo.intercept_)
    print("Error cuadrático medio (MSE):", mean_squared_error(y_test, y_pred))
    print("Coeficiente de determinación R^2:", r2)
    return r2

# ------------------ Bloque principal: carga y ejecución ------------------
# Ajusta estas rutas/variables a tu entorno
base_path = r"C:\Users\JJ\Documents\Mineria_Datos_AlgGenetico\Datasets"
file_name = "AirQuality_imputed_iterative.csv"
file_path = os.path.join(base_path, file_name)
columna_y = 'CO(GT)'

# Cargar datos
if file_name.endswith('.xlsx'):
    df = pd.read_excel(file_path)
elif file_name.endswith('.csv'):
    df = pd.read_csv(file_path)
else:
    raise ValueError("El archivo debe ser .csv o .xlsx")

if columna_y not in df.columns:
    raise ValueError(f"La columna '{columna_y}' no se encuentra en el archivo.")

y = df[columna_y]
X = df.drop(columns=[columna_y])

p_original = len(X.columns)
n_observations = len(df)
num_individuos = 300   # para pruebas usa población más pequeña si quieres (300 es costoso)
# (opcional) r2 del modelo completo, para diagnóstico
r2_full = regresion_lineal_multiple(df, X, y)
print(f"R² (todas las variables): {r2_full:.6f}")

# Inicializar y evaluar población con la función corregida
individuos_info = poblacion_inicial_y_aptitud(
    X, y, p_original, num_individuos, 
    test_size=0.2, random_state=42
)

# Ejecutar AG (nota: esto ajusta y evalúa modelos para cada individuo en cada generación => costoso)
mejores_soluciones = ejecutar_algoritmo_genetico(individuos_info, X, y, n_observations, num_individuos, pc=0.75, pm=0.01, n=30, test_size=0.2, random_state=42)

# Mejor solución global entre generacion
mejor_solucion = max(mejores_soluciones, key=lambda x: x["Ganancia"])
print("\nMejor solución optima:")
print(f"Individuo: {mejor_solucion['Individuo']}")
print(f"Vector binario: {mejor_solucion['Vector binario']}")
print(f"R2: {mejor_solucion['Peso_suma']}")
adj_str = f"{mejor_solucion['Ganancia']:.6f}" if np.isfinite(mejor_solucion['Ganancia']) else str(mejor_solucion['Ganancia'])
print(f"Adjusted R2 : {adj_str}")

print("\nAdjusted R2 de cada mejor individuo en esta corrida:")
for solucion in mejores_soluciones:
    adj = solucion['Ganancia']
    print(f"Individuo {solucion['Individuo']}: Adjusted R2 = {adj if np.isfinite(adj) else adj}")

# ------------------ BLOQUE: comparación, gráficas y resultados finales ------------------
# ------------------ BLOQUE: comparación, gráficas y resultados finales ------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Diccionario de traducción de variables a español
variable_descripciones = {
    "CO(GT)": "CO (monóxido de carbono medido)",
    "PT08.S1(CO)": "Sensor CO",
    "NMHC(GT)": "Hidrocarburos no metánicos (medidos)",
    "C6H6(GT)": "Benceno (medido)",
    "PT08.S2(NMHC)": "Sensor hidrocarburos no metánicos",
    "NOx(GT)": "NOx (medido)",
    "PT08.S3(NOx)": "Sensor NOx",
    "NO2(GT)": "Dióxido de nitrógeno (medido)",
    "PT08.S4(NO2)": "Sensor NO2",
    "PT08.S5(O3)": "Sensor O3",
    "T": "Temperatura",
    "RH": "Humedad relativa",
    "AH": "Humedad absoluta"
}

# Usar el mismo split para comparar modelos
X_train_all, X_test_all, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_test = len(y_test)

# Función para formatear ecuación
def format_equation(intercept, coefs, feature_names, max_terms=8):
    terms = [f"{coef:.4f}*{name}" for coef, name in zip(coefs, feature_names)]
    if len(terms) > max_terms:
        terms_short = terms[:max_terms] + [f"...(+{len(terms)-max_terms} más)"]
    else:
        terms_short = terms
    return "y = {:.4f} + ".format(intercept) + " + ".join(terms_short)

# 1) Modelo inicial: TODAS las variables
modelo_all = LinearRegression()
modelo_all.fit(X_train_all, y_train)
y_pred_all = modelo_all.predict(X_test_all)
r2_inicial = r2_score(y_test, y_pred_all)
p_all = X_train_all.shape[1]
adj_r2_inicial = (1 - (1 - r2_inicial) * (n_test - 1) / (n_test - p_all - 1)) if (n_test - p_all - 1) > 0 else np.nan
eq_all = format_equation(modelo_all.intercept_, modelo_all.coef_, X.columns.tolist(), max_terms=8)

# 2) Modelo final: variables seleccionadas
vector_final = mejor_solucion["Vector binario"]
variables_seleccionadas = [col for col, bit in zip(X.columns, vector_final) if int(bit) == 1]

if len(variables_seleccionadas) == 0:
    print("⚠ El vector binario final NO seleccionó ninguna variable.")
    r2_final, adj_r2_final = 0.0, np.nan
else:
    X_train_sel = X_train_all[variables_seleccionadas]
    X_test_sel  = X_test_all[variables_seleccionadas]

    modelo_sel = LinearRegression()
    modelo_sel.fit(X_train_sel, y_train)
    y_pred_sel = modelo_sel.predict(X_test_sel)

    r2_final = r2_score(y_test, y_pred_sel)
    p_sel = X_train_sel.shape[1]
    adj_r2_final = (1 - (1 - r2_final) * (n_test - 1) / (n_test - p_sel - 1)) if (n_test - p_sel - 1) > 0 else np.nan
    eq_sel = format_equation(modelo_sel.intercept_, modelo_sel.coef_, variables_seleccionadas, max_terms=12)

# ------------------ Graficar en paralelo ------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: todas las variables
axes[0].scatter(y_test, y_pred_all, alpha=0.6)
minv = min(y_test.min(), y_pred_all.min()); maxv = max(y_test.max(), y_pred_all.max())
axes[0].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
axes[0].set_xlabel("Valor real")
axes[0].set_ylabel("Valor predicho")
axes[0].set_title("Modelo con TODAS las variables")
axes[0].text(0.02, 0.98, f"R² = {r2_inicial:.4f}\nAdj R² = {adj_r2_inicial:.4f}", 
             transform=axes[0].transAxes, va="top", bbox=dict(facecolor='white', alpha=0.8))

# Gráfico 2: variables seleccionadas
if len(variables_seleccionadas) > 0:
    axes[1].scatter(y_test, y_pred_sel, alpha=0.6)
    minv = min(y_test.min(), y_pred_sel.min()); maxv = max(y_test.max(), y_pred_sel.max())
    axes[1].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
    axes[1].set_xlabel("Valor real")
    axes[1].set_ylabel("Valor predicho")
    axes[1].set_title("Modelo con VARIABLES SELECCIONADAS (AG)")
    axes[1].text(0.02, 0.98, f"R² = {r2_final:.4f}\nAdj R² = {adj_r2_final:.4f}", 
                 transform=axes[1].transAxes, va="top", bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# ------------------ Resumen ------------------
print("\n===== RESUMEN FINAL =====")
print(f"Total de variables iniciales: {p_all}")
print(f"Total de variables seleccionadas: {len(variables_seleccionadas)}")
print(f"Se redujo de {p_all} a {len(variables_seleccionadas)} variables.\n")

print(f"R² Inicial (todas las variables): {r2_inicial:.6f}")
print(f"Adjusted R² Inicial: {adj_r2_inicial:.6f}")

if len(variables_seleccionadas) == 0:
    print("R² Final: 0.0")
    print("Adjusted R² Final: NaN")
else:
    print(f"R² Final: {r2_final:.6f}")
    print(f"Adjusted R² Final: {adj_r2_final:.6f}")

print("\nVariables seleccionadas por el AG:")
if variables_seleccionadas:
    for v in variables_seleccionadas:
        print(f" - {v}: {variable_descripciones.get(v, 'Descripción no disponible')}")
else:
    print(" ⚠ Ninguna variable seleccionada.")