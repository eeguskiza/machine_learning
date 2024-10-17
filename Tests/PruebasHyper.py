import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from termcolor import colored

# Cargar el dataset limpio
energy_cleaned = pd.read_csv("\\Users\\eegus\\Desktop\\ML\\Energy\\cleaned.csv")

# Convertir variables categóricas en variables dummy
energy_cleaned = pd.get_dummies(energy_cleaned, drop_first=True)

# Seleccionar las características (features) y la variable objetivo (target)
X = energy_cleaned[['year', 'co2_emissions', 'latitude']] #Variables independientes
y = energy_cleaned['renewable_share'] #Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir los hiperparámetros a probar
n_estimators = [100, 200, 500]
max_depth = [5, 10, None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
max_features = [None, 'sqrt', 'log2']
bootstrap = [True, False]

# Inicializar archivo CSV para resultados
results_file = "\\Users\\eegus\\Desktop\\ML\\Energy\\hyperparameter_results.csv"
with open(results_file, 'w') as f:
    f.write("n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features,bootstrap,mse,r2,precise_predictions\n")

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

    # Guardar los resultados si la precisión supera el 70%
    if precise_predictions > 70:
        with open(results_file, 'a') as f:
            f.write(f"{n_est},{max_d},{min_split},{min_leaf},{max_feat},{boot},{mse},{r2},{precise_predictions}\n")

print("Se han probado todas las combinaciones de hiperparámetros y los resultados se han guardado en 'hyperparameter_results.csv'.")