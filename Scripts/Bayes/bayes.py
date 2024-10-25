from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Cargar los datos
data = pd.read_csv('cleaned.csv')

# Convertir 'co2_emissions' a clases categóricas
bins = [0, 50000, 100000, data['co2_emissions'].max()]
labels = ['Bajo', 'Medio', 'Alto']
data['co2_emissions_class'] = pd.cut(data['co2_emissions'], bins=bins, labels=labels)

# Remover filas con NaN en la variable objetivo
data = data.dropna(subset=['co2_emissions_class'])

# Seleccionar todas las columnas excepto la variable objetivo
feature_columns = [
    'renewable_capacity_pc', 'renewable_share', 'low_carbon_elec', 
    'gdp_growth', 'gdp_pc', 'energy_pc', 'access_clean_fuels', 'latitude', 'longitude'
]

# Variables para almacenar el mejor modelo y su precisión
best_accuracy = 0
best_features = None
best_model = None

# Probar diferentes combinaciones de características
for i in range(1, len(feature_columns) + 1):
    for combo in combinations(feature_columns, i):
        # Crear dataset de características para esta combinación
        X = data[list(combo)]
        y = data['co2_emissions_class']

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Hacer predicciones y evaluar precisión
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Almacenar el modelo si tiene la mejor precisión hasta ahora
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = combo
            best_model = model

# Imprimir el mejor conjunto de características
print(f"Best features found: {best_features}")
print("Testing hyperparameter combinations with best features...")

# Ajustar hiperparámetros con GridSearchCV
param_grid = {
    'var_smoothing': np.logspace(0, -9, num=10)
}

grid_search = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train[list(best_features)], y_train)

# Mejor modelo y precisión después de ajuste de hiperparámetros
best_hyper_model = grid_search.best_estimator_
y_pred_best = best_hyper_model.predict(X_test[list(best_features)])
best_hyper_accuracy = accuracy_score(y_test, y_pred_best)

print("Best hyperparameters:", grid_search.best_params_)
print(f"Accuracy with best hyperparameters: {best_hyper_accuracy:.2f}")
