import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Define paths
data_folder = "/home/liam-walker/Documents/IBANQ/Price Predict/FuelPricePredict/Data/"
data_folder = os.path.expanduser(data_folder)
input_path = os.path.join(data_folder, "combined_fuel_prices_long.csv")
output_path = os.path.join(data_folder, "processed_fuel_prices.csv")

# Load merged dataset
df = pd.read_csv(input_path)
df['Date'] = pd.to_datetime(df['Date'])

# Feature engineering
df['Lag_Fuel_Price'] = df.groupby('Zone')['Fuel Price'].shift(1)  # Previous month's price
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Encode Zone
df = pd.get_dummies(df, columns=['Zone'], drop_first=True)

# Drop rows with NaN (from lagging)
df = df.dropna()

# Define features and target
feature_cols = ['Exchange Rate', 'Crude Oil ($/bbl)', 'Lag_Fuel_Price', 'Month', 'Year'] + [col for col in df.columns if col.startswith('Zone_')]
X = df[feature_cols]
y = df['Fuel Price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed data with all columns for Superset
df.to_csv(output_path, index=False)
print(f"Processed dataset saved to: {output_path}")
print(f"Features: {feature_cols}")
print(f"Total rows after preprocessing: {len(df)}")