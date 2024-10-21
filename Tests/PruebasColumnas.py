import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from termcolor import colored
import random

# Cargar el dataset limpio
energy_cleaned = pd.read_csv("cleaned.csv")

# Definir la columna objetivo (deberás cambiar 'columna_objetivo' por la columna que deseas predecir)
column_to_predict = 'co2_emissions'  # Cambia esto según la columna que deseas predecir
y = energy_cleaned[column_to_predict]

# Convertir las variables categóricas en variables dummy
energy_cleaned = pd.get_dummies(energy_cleaned, drop_first=True)

# Lista de todas las columnas disponibles (después de convertir a dummy)
all_columns = energy_cleaned.columns.tolist()

# Quitar la columna objetivo de las columnas disponibles para las combinaciones de predictores
all_columns.remove(column_to_predict)

# Inicializar la mejor combinación
best_combination = None
best_precise_predictions = 0
best_result_str = ""

# Definir el número de combinaciones aleatorias a probar por tamaño
sample_size = 20  # Reducido para mayor eficiencia

# Iterar sobre combinaciones de columnas predictoras de tamaños diferentes
for i in range(1, min(5, len(all_columns) + 1)):  # Limitar el tamaño de las combinaciones a probar
    combinations_list = list(combinations(all_columns, i))
    if len(combinations_list) > sample_size:
        combinations_list = random.sample(combinations_list, sample_size)

    for combination in combinations_list:
        # Seleccionar las características (features)
        X = energy_cleaned[list(combination)]  # Variables independientes

        # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Hacer predicciones con el conjunto de prueba
        y_pred = model.predict(X_test)

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

        # Actualizar la mejor combinación si es mejor que la anterior
        if precise_predictions > best_precise_predictions:
            best_precise_predictions = precise_predictions
            best_combination = combination
            best_result_str = f"Mejor combinación actual - Características: {combination}, MSE: {mse}, R2: {r2}, Precisión dentro del 5%: {precise_predictions}%"

# Imprimir la mejor combinación resaltada
print("\n***************** MEJOR COMBINACIÓN *****************")
print(colored(best_result_str, 'cyan', attrs=['bold']))
print("****************************************************")