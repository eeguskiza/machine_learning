import pandas as pd

# Cargar los datos
energy = pd.read_csv("/Users/eeguskiza/Documents/Deusto/github/machine_learning/energy.csv")

# Diccionario para renombrar las columnas a nombres más cortos
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
    'Density\n(P/Km2)': 'density',
    'Land Area(Km2)': 'land_area',
    'Latitude': 'latitude',
    'Longitude': 'longitude'
}

# Renombrar las columnas del DataFrame
energy.rename(columns=column_rename_dict, inplace=True)

# Eliminar filas con valores faltantes
energy_cleaned = energy.dropna()

# Exportar el DataFrame limpio a un nuevo archivo CSV
energy_cleaned.to_csv("/Users/eeguskiza/Documents/Deusto/ML/Energy/cleaned.csv", index=False)

print("El dataset ha sido limpiado y exportado como 'cleaned.csv'.")