import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

# Cargar el dataset ya renombrado
data = pd.read_csv('/Users/eeguskiza/Documents/Deusto/github/machine_learning/cleaned.csv')

# Convertir 'entity' a valores numéricos
label_encoder = LabelEncoder()
data['entity'] = label_encoder.fit_transform(data['entity'])

# Remover comas de las columnas numéricas y convertir a float
data = data.replace(',', '', regex=True).astype(float)

# Variables predictoras y variable objetivo
X = data.drop(columns=['renewable_share'])  # Cambia 'renewable_share' según tu preferencia
y = np.where(data['renewable_share'] > 50, 1, 0)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo Random Forest con los mejores parámetros
best_rf_model = RandomForestClassifier(
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=150,
    random_state=42
)

# Entrenar el modelo
best_rf_model.fit(X_train, y_train)

# Hacer predicciones y calcular precisión
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", accuracy)