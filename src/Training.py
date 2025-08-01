import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Define paths
data_folder = "/home/liam-walker/Documents/IBANQ/Price Predict/FuelPricePredict/Data/"
data_folder = os.path.expanduser(data_folder)
input_path = os.path.join(data_folder, "processed_fuel_prices.csv")

# Load processed data
df = pd.read_csv(input_path)
df['Date'] = pd.to_datetime(df['Date'])

# Define features
feature_cols = ['Exchange Rate', 'Crude Oil ($/bbl)', 'Lag_Fuel_Price', 'Month', 'Year'] + [col for col in df.columns if col.startswith('Zone_')]
X = df[feature_cols]
y = df['Fuel Price']

# Time-based split (train: 2019–2023, test: 2024–2025)
train_df = df[df['Date'] < '2024-01-01']
test_df = df[df['Date'] >= '2024-01-01']
X_train = train_df[feature_cols]
y_train = train_df['Fuel Price']
X_test = test_df[feature_cols]
y_test = test_df['Fuel Price']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")

# Save model and scaler
model_path = os.path.join(data_folder, "fuel_price_model.joblib")
scaler_path = os.path.join(data_folder, "scaler.joblib")
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")