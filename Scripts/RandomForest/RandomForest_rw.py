import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Load the dataset
energy = pd.read_csv('/Users/eeguskiza/Documents/Deusto/github/machine_learning/cleaned.csv')


X = energy[['year', 'co2_emissions', 'latitude']] #Variables independientes
y = energy['renewable_share'] #Variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parámetros: (100, None, 2, 1, None, False), Precisión dentro del 5%: 75.36231884057972% #########
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    bootstrap=False,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test) #Predicciones en el conjunto de prueba

#Umbral de aceptacion
thersold_percentage = 0.05

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
predictions_score = np.mean(np.abs((y_test - y_pred) / y_test) < thersold_percentage) * 100

print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")
print(f"Prediction accuracy score: {predictions_score}")
