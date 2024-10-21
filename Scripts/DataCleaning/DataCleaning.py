import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
energy = pd.read_csv("/Users/eeguskiza/Documents/Deusto/github/machine_learning/energy.csv")

# Plot initial distributions to understand the data
plt.figure(figsize=(14, 8))
sns.histplot(energy['Access to electricity (% of population)'], kde=True, color='blue', label='Access to Electricity')
sns.histplot(energy['Renewable energy share in the total final energy consumption (%)'], kde=True, color='green', label='Renewable Share')
plt.legend()
plt.title('Initial Distribution of Energy Features')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Rename the columns to shorter names
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

# Drop rows with missing values
energy.dropna(inplace=True)

# Drop duplicate rows
energy.drop_duplicates(inplace=True)

# Cap the outliers instead of removing them using the 1st and 99th percentile
numerical_cols = [
    'access_electricity', 'renewable_share', 'fossil_fuels_elec', 'nuclear_elec', 'renewables_elec',
    'low_carbon_elec', 'energy_pc', 'energy_intensity', 'co2_emissions', 'renewables_pc',
    'gdp_growth', 'gdp_pc', 'density', 'land_area'
]

# Remove columns that may not exist
numerical_cols = [col for col in numerical_cols if col in energy.columns]

for col in numerical_cols:
    lower_percentile = energy[col].quantile(0.01)
    upper_percentile = energy[col].quantile(0.99)
    energy[col] = energy[col].clip(lower=lower_percentile, upper=upper_percentile)

# Normalize numerical data for Machine Learning
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
energy[numerical_cols] = scaler.fit_transform(energy[numerical_cols])

# Export cleaned DataFrame to a new CSV file
energy.to_csv("cleaned.csv", index=False)

# Plot final distributions to compare with the initial data
plt.figure(figsize=(14, 8))
sns.histplot(energy['access_electricity'], kde=True, color='blue', label='Access to Electricity')
sns.histplot(energy['renewable_share'], kde=True, color='green', label='Renewable Share')
plt.legend()
plt.title('Final Distribution of Energy Features (After Cleaning)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

print("El dataset ha sido limpiado, normalizado y exportado como 'cleaned.csv'.")
