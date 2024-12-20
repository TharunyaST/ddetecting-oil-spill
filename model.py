import pandas as pd
import numpy as np
import folium
from geopy.distance import geodesic

# Vectorized haversine function for distance calculations
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Earth's radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Step 1: Load Data
try:
    data = pd.read_csv(r'G:\SIH_FINAL_AIS_SATE\intergration\dataset.csv')
    # data = data.iloc[390:491]
    tugboats_data = pd.read_csv(r'G:\SIH_FINAL_AIS_SATE\intergration\tug_boats.csv')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Convert BaseDateTime to datetime format and sort
data['BaseDateTime'] = pd.to_datetime(data['BaseDateTime'])
data.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)

# Step 2: Feature Engineering
def calculate_features(df):
    # Calculate time difference
    df['delta_time'] = df['BaseDateTime'].diff().dt.total_seconds().fillna(1)

    # Calculate differences
    df['delta_COG'] = df['COG'].diff().fillna(0)
    df['delta_SOG'] = df['SOG'].diff().fillna(0)
    df['delta_Heading'] = df['Heading'].diff().fillna(0)

    # Calculate distance traveled using haversine
    df['prev_LAT'] = df['LAT'].shift()
    df['prev_LON'] = df['LON'].shift()
    df['distance_traveled'] = haversine(
        df['LAT'], df['LON'], df['prev_LAT'], df['prev_LON']
    ).fillna(0)

    # Rolling statistics
    df['rolling_mean_SOG'] = df['SOG'].rolling(window=5).mean().fillna(method='bfill')
    df['rolling_std_SOG'] = df['SOG'].rolling(window=5).std().fillna(0)
    return df

# Apply feature engineering
data = data.groupby('MMSI', group_keys=False).apply(calculate_features)

# Step 3: Adaptive Thresholds
def calculate_thresholds(df):
    df['COG_threshold'] = df['COG'].mean() + 3 * df['COG'].std()
    df['SOG_threshold'] = df['SOG'].mean() + 3 * df['SOG'].std()
    df['Heading_threshold'] = df['Heading'].mean() + 3 * df['Heading'].std()
    df['distance_threshold'] = df['distance_traveled'].mean() + 3 * df['distance_traveled'].std()
    return df

# Apply thresholds
data = data.groupby('MMSI', group_keys=False).apply(calculate_thresholds)

# Step 4: Anomaly Detection
def detect_anomalies(row):
    reasons = []
    if abs(row['delta_COG']) > row['COG_threshold']:
        reasons.append("Sharp Turn")
    if abs(row['delta_SOG']) > row['SOG_threshold']:
        reasons.append("Speed Change")
    if row['distance_traveled'] > row['distance_threshold']:
        reasons.append("Unusual Distance Traveled")

    if len(reasons) >= 2:  # Anomaly if at least two criteria met
        return True, ", ".join(reasons)
    return False, None

data[['anomaly', 'anomaly_reason']] = data.apply(detect_anomalies, axis=1, result_type='expand')

# Step 5: Save Anomalies to CSV
anomaly_data = data[data['anomaly']].dropna(subset=['LAT', 'LON'])

# Save anomalies to CSV
if not anomaly_data.empty:
    anomaly_data.to_csv(r'G:\SIH_FINAL_AIS_SATE\intergration\detected_anomalies_model.csv', index=False)
    print("Anomalies saved to 'detected_anomalies.csv'.")
else:
    print("No anomalies detected.")

# # Step 6: Visualize Anomalies
# if not anomaly_data.empty:
#     avg_lat, avg_lon = anomaly_data['LAT'].mean(), anomaly_data['LON'].mean()
#     m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)

#     for _, row in anomaly_data.iterrows():
#         folium.Marker(
#             location=[row['LAT'], row['LON']],
#             popup=(f"MMSI: {row['MMSI']}<br>Time: {row['BaseDateTime']}<br>"
#                    f"Reason: {row['anomaly_reason']}"),
#             icon=folium.Icon(color='red')
#         ).add_to(m)

#     # Save map
#     m.save('optimized_anomaly_map.html')
#     print("Anomaly map saved as 'optimized_anomaly_map.html'.")
# else:
#     print("No anomalies detected for visualization.")
