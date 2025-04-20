import pandas as pd
from datetime import timedelta
import numpy as np

df = pd.read_csv(r"C:\Users\LENOVO\Downloads\amazon_delivery.csv",na_values=['NaN ', 'NULL', 'null', ''])

df['Agent_Rating'].fillna(df['Agent_Rating'].median(), inplace=True)

df['Weather'] = df.groupby('Area')['Weather'].transform(lambda x: x.fillna(x.mode()[0]))

df['Traffic'] = df.groupby('Area')['Traffic'].transform(lambda x: x.fillna(x.mode()[0]))

# Convert time columns to datetime by combining with Order_Date
df['Order_DateTime'] = pd.to_datetime(df['Order_Date'].astype(str))

df['Order_DateTime_Order'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Order_Time'].astype(str), format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['Order_DateTime_Pickup'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Pickup_Time'].astype(str), format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Calculate average time difference per Area (in minutes)
df_non_null = df.dropna(subset=['Order_DateTime_Order', 'Order_DateTime_Pickup']).copy()
df_non_null['Time_Diff'] = (df_non_null['Order_DateTime_Pickup'] - df_non_null['Order_DateTime_Order']).dt.total_seconds() / 60

# Remove negative diffs before averaging
df_non_null = df_non_null[df_non_null['Time_Diff'] >= 0]

# Calculate average time diff per Area
area_time_diff = df_non_null.groupby('Area')['Time_Diff'].mean().to_dict()

# Impute missing Order_Time based on Pickup_Time and avg area diff (rounded)
for idx, row in df[df['Order_Time'].isnull()].iterrows():
    if pd.notnull(row['Pickup_Time']):
        avg_diff = area_time_diff.get(row['Area'], 15)  # Default 15 mins if not found
        avg_diff = max(avg_diff, 0)  # Avoid negative diffs
        pickup_datetime = pd.to_datetime(f"{row['Order_Date']} {row['Pickup_Time']}", format='%Y-%m-%d %H:%M:%S', errors='coerce')
        if pd.notnull(pickup_datetime):
            # Subtract rounded avg_diff in minutes
            new_order_datetime = pickup_datetime - timedelta(minutes=round(avg_diff))
            # Update Order_Time with HH:MM:SS string
            df.at[idx, 'Order_Time'] = new_order_datetime.strftime('%H:%M:%S')

# Clean up if needed
df.drop(columns=['Order_DateTime', 'Order_DateTime_Order', 'Order_DateTime_Pickup'], inplace=True)

# 1️⃣ Remove invalid lat/lon values outside of possible ranges
mask_invalid_coords = (
    (df['Store_Latitude'].abs() > 90) | (df['Drop_Latitude'].abs() > 90) |
    (df['Store_Longitude'].abs() > 180) | (df['Drop_Longitude'].abs() > 180)
)
df_clean = df[~mask_invalid_coords].copy()

# 2️⃣ Remove rows where Store/Drop coords are zero (unless valid)
mask_zero_coords = (
    (df_clean['Store_Latitude'] == 0) | (df_clean['Drop_Latitude'] == 0) |
    (df_clean['Store_Longitude'] == 0) | (df_clean['Drop_Longitude'] == 0)
)
df_clean = df_clean[~mask_zero_coords]

# 3️⃣ Flag unreasonable distances (> 50 km for same-city delivery)
df_clean['Lat_Diff'] = abs(df_clean['Store_Latitude'] - df_clean['Drop_Latitude'])
df_clean['Lon_Diff'] = abs(df_clean['Store_Longitude'] - df_clean['Drop_Longitude'])

# You can tweak this threshold — assuming ~0.5 deg = ~50km approx.
mask_far_deliveries = (df_clean['Lat_Diff'] > 0.5) | (df_clean['Lon_Diff'] > 0.5)

# View bad deliveries
bad_deliveries = df_clean[mask_far_deliveries]

# Option 1: Drop them
df_clean = df_clean[~mask_far_deliveries]

# 4️⃣ Drop helper columns if not needed
df_clean.drop(columns=['Lat_Diff', 'Lon_Diff'], inplace=True)

df_clean.to_csv("Cleaned_Dataset.csv",index=False)

df_clean.isnull().sum()

print(df_clean.shape)

