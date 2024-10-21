import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
energy = pd.read_csv("/Users/eeguskiza/Documents/Deusto/github/machine_learning/energy.csv")

# Plotear distribuciones iniciales para entender los datos
numerical_cols = energy.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(20, 12))
plt.suptitle('Distribución Inicial de las Características Numéricas', fontsize=20)
num_plots_per_row = 3
for i, col in enumerate(numerical_cols, 1):
    plt.subplot((len(numerical_cols) + num_plots_per_row - 1) // num_plots_per_row, num_plots_per_row, i)
    sns.histplot(energy[col], kde=True)
    plt.title(col)
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Renombrar columnas a nombres más cortos
column_rename_dict = {
    'Entity': 'entity',
    'Year': 'year',
    'Access to electricity (% of population)': 'access_electricity',
    'Access to clean fuels for cooking': 'access_clean_fuels',
    'Renewable-electricity-generating-capacity-per-capita': 'renewable_capacity_pc',
    'Financial flows to developing countries (US $)': 'financial_flows',
    'Renewable energy share in the total final energy consumption (%)': 'renewable_share',
    'Electricity from fossil fuels (TWh)': 'fossil_fuels_elec',
    'Electricity from nuclear (TWh)': 'nuclear_elec',
    'Electricity from renewables (TWh)': 'renewables_elec',
    'Low-carbon electricity (% electricity)': 'low_carbon_elec',
    'Primary energy consumption per capita (kWh/person)': 'energy_pc',
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)': 'energy_intensity',
    'Value_co2_emissions_kt_by_country': 'co2_emissions',
    'Renewables (% equivalent primary energy)': 'renewables_pc',
    'gdp_growth': 'gdp_growth',
    'gdp_per_capita': 'gdp_pc',
    'Density (P/Km2)': 'density',
    'Land Area(Km2)': 'land_area',
    'Latitude': 'latitude',
    'Longitude': 'longitude'
}

energy.rename(columns=column_rename_dict, inplace=True)

# Eliminar filas con valores faltantes
energy.dropna(inplace=True)

# Eliminar filas duplicadas
energy.drop_duplicates(inplace=True)

# Limitar outliers usando percentil 1 y 99
numerical_cols = [col for col in numerical_cols if col in energy.columns]
for col in numerical_cols:
    lower_percentile = energy[col].quantile(0.01)
    upper_percentile = energy[col].quantile(0.99)
    energy[col] = energy[col].clip(lower=lower_percentile, upper=upper_percentile)

# Normalizar los datos numéricos para Machine Learning
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
energy[numerical_cols] = scaler.fit_transform(energy[numerical_cols])

# Exportar el DataFrame limpio a un nuevo archivo CSV
energy.to_csv("cleaned.csv", index=False)

# Plotear distribuciones finales para comparar con los datos iniciales
plt.figure(figsize=(20, 12))
plt.suptitle('Distribución Final de las Características Numéricas (Después de Limpiar)', fontsize=20)
num_plots_per_row = 3
for i, col in enumerate(numerical_cols, 1):
    plt.subplot((len(numerical_cols) + num_plots_per_row - 1) // num_plots_per_row, num_plots_per_row, i)
    sns.histplot(energy[col], kde=True)
    plt.title(col)
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("El dataset ha sido limpiado, normalizado y exportado como 'cleaned.csv'.")