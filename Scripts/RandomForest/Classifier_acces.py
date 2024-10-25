import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

# Cargar el dataset ya renombrado
data = pd.read_csv('/Users/eeguskiza/Documents/Deusto/github/machine_learning/cleaned.csv')

# Convertir 'entity' a valores numéricos
label_encoder = LabelEncoder()
data['entity'] = label_encoder.fit_transform(data['entity'])

# Convertir columnas con comas a formato numérico adecuado
data = data.replace(',', '', regex=True).astype(float)

# Variables predictoras y variable objetivo
X = data.drop(columns=['renewable_share'])  # Cambia 'renewable_share' por la variable que desees
y = np.where(data['renewable_share'] > 50, 1, 0)  # Clasificación binaria basada en la nueva variable

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir la cuadrícula de hiperparámetros para Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configurar y ajustar el modelo Random Forest con StratifiedKFold
rf_model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluar el mejor modelo en el conjunto de prueba
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Best parameters:", grid_search.best_params_)
print("Model accuracy:", accuracy)
