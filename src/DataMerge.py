import pandas as pd
import glob
import os

# Define the path to the folder containing zone CSV files
data_folder = "/home/liam-walker/Documents/IBANQ/Zone_MAP_Proj/Datasets/ZonePrice/ZoneExch/"

# Expand the home directory (~) to full path
data_folder = os.path.expanduser(data_folder)

# Get list of all CSV files in the folder
zone_files = glob.glob(os.path.join(data_folder, "*_FuelExch.csv"))

# Initialize an empty list to store dataframes
dfs = []

# Load Exchange Rate and Crude Oil ($/bbl) from the first file
base_df = pd.read_csv(zone_files[0], usecols=["Date", "Exchange Rate", "Crude Oil ($/bbl)"])
base_df['Date'] = pd.to_datetime(base_df['Date'])

# Loop through each CSV file to extract Zone and Fuel Price
for file in zone_files:
    df = pd.read_csv(file, usecols=["Date", "Zone", "Fuel Price"])
    df['Date'] = pd.to_datetime(df['Date'])
    # Merge with base features
    df = df.merge(base_df, on="Date", how="left")
    dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Sort by Date and Zone
combined_df = combined_df.sort_values(['Date', 'Zone'])

# Verify
print(f"Total rows: {len(combined_df)}")
print(f"Unique zones: {combined_df['Zone'].unique()}")
print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
print(combined_df.head())

# Save the merged dataset
output_path = os.path.join(data_folder, "combined_fuel_prices_long.csv")
combined_df.to_csv(output_path, index=False)
print(f"Merged dataset saved to: {output_path}")