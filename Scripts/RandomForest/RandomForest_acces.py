import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from termcolor import colored

# Cargar el dataset
energy = pd.read_csv('/Users/eeguskiza/Documents/Deusto/github/machine_learning/cleaned.csv')

# Definir las características y la variable objetivo
X = energy[['year', 'renewable_share', 'energy_pc', 'longitude']]  # Variables independientes
y = energy['access_electricity']  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mejor combinación de hiperparámetros según el análisis previo
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    bootstrap=True,
    random_state=42
)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Definir un umbral de aceptación
threshold_percentage = 0.05

# Calcular el error cuadrático medio, el coeficiente de determinación R^2 y la precisión
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
precise_predictions = np.mean(np.abs((y_test - y_pred) / y_test) < threshold_percentage) * 100

# Imprimir los resultados
print(colored("***************** RESULTADOS *****************", 'cyan', attrs=['bold']))
print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")
print(f"Prediction accuracy score (within 5% threshold): {precise_predictions}%")
print(colored("***********************************************", 'cyan', attrs=['bold']))
