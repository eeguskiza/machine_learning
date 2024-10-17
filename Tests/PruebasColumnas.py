import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from termcolor import colored

# Cargar el dataset limpio
energy_cleaned = pd.read_csv("\\Users\\eegus\\Desktop\\ML\\Energy\\cleaned.csv")

# Convertir variables categóricas en variables dummy
energy_cleaned = pd.get_dummies(energy_cleaned, drop_first=True)

# Inicializar archivo CSV para resultados
results_file = "model_results.csv"
with open(results_file, 'w') as f:
    f.write("features,mse,r2,precise_predictions\n")

# Lista de todas las columnas disponibles
all_columns = energy_cleaned.columns.tolist()

# Quitar columnas que no tienen sentido como predictoras
all_columns.remove('co2_emissions')  # No podemos predecir con la misma columna objetivo

# Iterar sobre todas las combinaciones posibles de columnas predictoras
for i in range(1, len(all_columns) + 1):
    for combination in combinations(all_columns, i):
        # Seleccionar las características (features) y la variable objetivo (target)
        X = energy_cleaned[list(combination)]  # Variables independientes
        y = energy_cleaned['co2_emissions']  # Variable objetivo

        # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Hacer predicciones con el conjunto de prueba
        y_pred = model.predict(X_test)

        # Calcular el porcentaje de precisión
        # Definir un umbral del 5% (puedes ajustarlo si lo deseas)
        threshold_percentage = 0.05  # Esto significa un 5% de diferencia permitida

        # Calcular el error cuadrático medio y el coeficiente de determinación R^2
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calcular el porcentaje de predicciones dentro del umbral del 5%
        precise_predictions = np.mean(np.abs((y_test - y_pred) / y_test) < threshold_percentage) * 100

        # Imprimir los resultados de cada combinación con colores según el rango de precisión
        if precise_predictions > 40:
            if 40 < precise_predictions <= 60:
                print(colored(f"Combinación: {combination}, Precisión dentro del 5%: {precise_predictions}%", 'yellow'))
            elif 60 < precise_predictions <= 70:
                print(colored(f"Combinación: {combination}, Precisión dentro del 5%: {precise_predictions}%", 'green'))
            elif 70 < precise_predictions <= 90:
                print(colored(f"######### Combinación: {combination}, Precisión dentro del 5%: {precise_predictions}% #########", 'magenta'))
            elif precise_predictions > 90:
                print(colored(f"Combinación: {combination}, Precisión dentro del 5%: {precise_predictions}%", 'white'))

        # Guardar los resultados si la precisión supera el 70%
        if precise_predictions > 70:
            with open(results_file, 'a') as f:
                f.write(f"{combination},{mse},{r2},{precise_predictions}\n")

print("Se han probado todas las combinaciones y los resultados se han guardado en 'model_results.csv'.")