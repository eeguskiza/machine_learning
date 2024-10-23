from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from itertools import combinations

# Cargar el dataset
dataset = pd.read_csv('/Users/eeguskiza/Documents/Deusto/github/machine_learning/cleaned.csv')

# Convertir columnas de tipo 'object' a valores numéricos
label_encoders = {}
for column in dataset.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])
    label_encoders[column] = le

# Definir la variable objetivo
y = np.where(dataset['access_electricity'] > 80, 1, 0)  # Clasificación binaria: acceso a electricidad mayor a 80%

# Inicializar variables para almacenar la mejor combinación y precisión
best_accuracy = 0
best_features = None
prueba_num = 1
max_pruebas = 1000

# Iterar sobre todas las combinaciones posibles de características (2 a la vez, hasta todas)
columns = dataset.columns.drop('access_electricity')
for r in range(2, len(columns) + 1):
    for combination in combinations(columns, r):
        if prueba_num > max_pruebas:
            break
        # Definir las variables predictoras
        X = dataset[list(combination)]
        
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
        
        # Imprimir el resultado de la prueba actual
        print(f'Prueba {prueba_num} ----------> Precisión: {accuracy:.2f}')
        prueba_num += 1
        
        # Actualizar la mejor combinación si la precisión es mayor
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = combination

    if prueba_num > max_pruebas:
        break

# Imprimir la mejor combinación de características y la precisión correspondiente
print(f'Mejor precisión del modelo: {best_accuracy:.2f} con las características: {best_features}')
