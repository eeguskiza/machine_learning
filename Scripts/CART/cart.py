from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Cargar el dataset
dataset = pd.read_csv('/Users/eeguskiza/Documents/Deusto/github/machine_learning/cleaned.csv')

# Convertir la columna 'entity' a valores numéricos
label_encoder = LabelEncoder()
dataset['entity'] = label_encoder.fit_transform(dataset['entity'])

# Variables predictoras y objetivo (ejemplo con columnas 'renewable_share' y 'gdp_growth')
X = dataset[['entity', 'year', 'latitude']]
y = np.where(dataset['access_electricity'] > 80, 1, 0)  # Clasificación binaria: acceso a electricidad mayor a 80%

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
dt_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = dt_model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')
