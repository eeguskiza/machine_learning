import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from termcolor import colored

# Definir las características y la variable objetivo
predictor_columns = ['year','fossil_fuels_elec', 'renewables_elec', 'gdp_growth', 'land_area', 'longitude']  # Cambia esto según tus necesidades
objective_column = 'co2_emissions'  # Cambia esto según tus necesidades

# Cargar el dataset limpio
energy_cleaned = pd.read_csv("/Users/eeguskiza/Documents/Deusto/github/machine_learning/cleaned.csv")

# Convertir variables categóricas en variables dummy
energy_cleaned = pd.get_dummies(energy_cleaned, drop_first=True)

# Verificar si las características seleccionadas existen en el dataframe
missing_features = [feature for feature in predictor_columns if feature not in energy_cleaned.columns]
if missing_features:
    raise ValueError(f"Las siguientes características no se encontraron en el dataset: {missing_features}")

# Seleccionar las características y la variable objetivo
X = energy_cleaned[predictor_columns]  # Variables independientes
y = energy_cleaned[objective_column]  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir los hiperparámetros a probar
n_estimators = [100, 200, 500, 1000]
max_depth = [5, 10, 15, 20, None]
min_samples_split = [2, 5, 10, 15]
min_samples_leaf = [1, 2, 4, 6]
max_features = [None, 'sqrt', 'log2']
bootstrap = [True, False]

# Inicializar la mejor combinación
best_params = None
best_precise_predictions = 0
best_result_str = ""

# Iterar sobre todas las combinaciones posibles de hiperparámetros
for params in product(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap):
    n_est, max_d, min_split, min_leaf, max_feat, boot = params

    # Crear y entrenar el modelo de Random Forest con la combinación de hiperparámetros
    model = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        max_features=max_feat,
        bootstrap=boot,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Hacer predicciones con el conjunto de prueba
    y_pred = model.predict(X_test)

    # Definir un umbral del 5% (puedes ajustarlo si lo deseas)
    threshold_percentage = 0.05

    # Calcular el error cuadrático medio, el coeficiente de determinación R^2 y la precisión
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    precise_predictions = np.mean(np.abs((y_test - y_pred) / y_test) < threshold_percentage) * 100

    # Imprimir los resultados de cada combinación con colores según el rango de precisión
    if precise_predictions > 40:
        if 40 < precise_predictions <= 60:
            print(colored(f"Parámetros: {params}, Precisión dentro del 5%: {precise_predictions}%", 'yellow'))
        elif 60 < precise_predictions <= 70:
            print(colored(f"Parámetros: {params}, Precisión dentro del 5%: {precise_predictions}%", 'green'))
        elif 70 < precise_predictions <= 90:
            print(colored(f"######### Parámetros: {params}, Precisión dentro del 5%: {precise_predictions}% #########", 'magenta'))
        elif precise_predictions > 90:
            print(colored(f"Parámetros: {params}, Precisión dentro del 5%: {precise_predictions}%", 'white'))

    # Actualizar la mejor combinación si es mejor que la anterior
    if precise_predictions > best_precise_predictions:
        best_precise_predictions = precise_predictions
        best_params = params
        best_result_str = f"Mejor combinación actual - Parámetros: {params}, MSE: {mse}, R2: {r2}, Precisión dentro del 5%: {precise_predictions}%"

# Imprimir la mejor combinación resaltada
print("\n***************** MEJOR COMBINACIÓN *****************")
print(colored(best_result_str, 'cyan', attrs=['bold']))
print("****************************************************")
