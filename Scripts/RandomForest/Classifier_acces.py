import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from termcolor import colored

# Cargar el dataset
energy = pd.read_csv('/Users/eeguskiza/Documents/Deusto/github/machine_learning/cleaned.csv')

# Definir las características y la variable objetivo
# Utilizando una selección más reducida de características
X = energy[['renewable_share', 'gdp_pc', 'latitude', 'longitude']]  # Variables independientes
y = energy['gdp_growth']  # Nueva variable objetivo, que no es categórica originalmente

# Convertir la variable objetivo en clases categóricas
# Clasificar como 1 si el crecimiento del PIB es positivo, de lo contrario 0
y = (y > 0).astype(int)

# Submuestreo de la clase mayoritaria para equilibrar las clases
# Obtener índices de las clases
class_0_indices = y[y == 0].index
class_1_indices = y[y == 1].index

# Balancear las clases tomando un número similar de muestras de cada clase
n_samples = min(len(class_0_indices), len(class_1_indices))
class_0_indices = np.random.choice(class_0_indices, n_samples, replace=False)
class_1_indices = np.random.choice(class_1_indices, n_samples, replace=False)

# Crear un nuevo dataset equilibrado
balanced_indices = np.concatenate([class_0_indices, class_1_indices])
X_balanced = X.loc[balanced_indices]
y_balanced = y.loc[balanced_indices]

# Dividir los datos en conjuntos de entrenamiento y prueba, con estratificación para mantener proporciones de clase
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Crear el modelo de Random Forest con los hiperparámetros adecuados
model = RandomForestClassifier(
    n_estimators=150,  # Número de árboles en el bosque
    max_depth=7,  # Reducir la profundidad para evitar sobreajuste
    min_samples_split=4,  # Aumentar el mínimo de muestras para dividir un nodo
    min_samples_leaf=2,  # Aumentar el mínimo de muestras en una hoja
    max_features='sqrt',  # Selección de características automáticas
    bootstrap=True,  # Utilizar bootstrap samples para construir árboles
    class_weight='balanced',  # Ajustar los pesos de las clases para manejar el desbalance
    random_state=42
)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Imprimir los resultados
print(colored("***************** RESULTADOS *****************", 'cyan', attrs=['bold']))
print(f"Accuracy score: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(colored("***********************************************", 'cyan', attrs=['bold']))

# Validación cruzada para evaluar el rendimiento
scores = cross_val_score(model, X_balanced, y_balanced, cv=5)
print(colored("\nCross-validation scores:", 'green'))
print(scores)
print(f"Average cross-validation score: {scores.mean()}")
