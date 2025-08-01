import pandas as pd
import numpy as np
import joblib
import os

# Define paths
data_folder = "/home/liam-walker/Documents/IBANQ/Price Predict/FuelPricePredict/Data/"
data_folder = os.path.expanduser(data_folder)
input_path = os.path.join(data_folder, "processed_fuel_prices.csv")
model_path = os.path.join(data_folder, "fuel_price_model.joblib")
scaler_path = os.path.join(data_folder, "scaler.joblib")
output_path = os.path.join(data_folder, "may_2025_predictions.csv")

# Load processed data and model
df = pd.read_csv(input_path)
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Get April 2025 data for Lag_Fuel_Price
april_2025 = df[df['Date'] == '2025-04-01']
zone_cols = [col for col in df.columns if col.startswith('Zone_')]
zones = [col.replace('Zone_', '') for col in zone_cols] + ['01A']  # Add reference zone (dropped in one-hot encoding)

# Create May 2025 features
may_2025_data = []
feature_cols = ['Exchange Rate', 'Crude Oil ($/bbl)', 'Lag_Fuel_Price', 'Month', 'Year'] + zone_cols
exchange_rate_may_2025 = 18.1087  # Replace with your forecasted value
crude_oil_may_2025 = 64.45  # Historical average or external estimate

for _, row in april_2025.iterrows():
    # Determine zone from one-hot encoded columns
    zone = '01A'  # Default if all zone_cols are 0
    for col in zone_cols:
        if row[col] == 1:
            zone = col.replace('Zone_', '')
            break
    # Create feature vector
    zone_values = [row[col] for col in zone_cols]
    features = [exchange_rate_may_2025, crude_oil_may_2025, row['Fuel Price'], 5, 2025] + zone_values
    may_2025_data.append((zone, features))

# Scale and predict
may_2025_features = np.array([features for _, features in may_2025_data])
may_2025_scaled = scaler.transform(may_2025_features)
predictions = model.predict(may_2025_scaled)

# Save predictions
results = pd.DataFrame({
    'Zone': [zone for zone, _ in may_2025_data],
    'Predicted_Fuel_Price': predictions
})
results.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")
print(results)