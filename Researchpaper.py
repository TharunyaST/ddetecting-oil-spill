import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy.spatial.distance import cdist
from scipy.stats import chi2
import matplotlib.pyplot as plt

data = pd.read_csv(r"G:\SIH_FINAL_AIS_SATE\intergration\dataset.csv")
data = data.iloc[:547466]

# Convert the 'BaseDateTime' column to datetime format
data["BaseDateTime"] = pd.to_datetime(data["BaseDateTime"], errors='coerce')
data = data[["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]]
data.sort_values(by=["MMSI", "BaseDateTime"], inplace=True)

basis_centers = np.array([[27.35, -96.15], [27.36, -96.16]])  # Example coordinates
weights = np.zeros(len(basis_centers))  # Initialize weights
learning_rate = 0.01
forgetting_factor = 0.9

# Basis Function Expansion
def basis_function_expansion(x, basis_centers, weights):
    distances = cdist([x[:2]], basis_centers, metric="euclidean").flatten()
    dynamics = np.exp(-distances)
    return weights[:, None] * dynamics

# Recursive Basis Function Weight Update
def update_basis_weights(weights, K, z, y_pred):
    residual = z - y_pred[:2]
    return weights + K @ residual

# State Transition with Basis Dynamics
def state_transition_with_basis(x, speed, course, time_diff, basis_centers, weights):
    g_k = basis_function_expansion(x, basis_centers, weights).sum(axis=0)
    course_rad = np.radians(course)
    speed_ms = speed * 0.51444
    distance = speed_ms * time_diff
    
    delta_lat = distance * np.cos(course_rad) / 111320
    delta_lon = distance * np.sin(course_rad) / (111320 * np.cos(np.radians(x[0])))
    
    return np.array([x[0] + delta_lat + g_k[0], x[1] + delta_lon + g_k[1], speed, course])

# Kalman Filter Steps
def kalman_filter_predict(x, P, F, Q):
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), F.T) + Q
    return x_pred, P_pred

def kalman_filter_update(x_pred, P_pred, z, H, R):
    y = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    K = np.dot(np.dot(P_pred, H.T), inv(S))
    x_updated = x_pred + np.dot(K, y)
    P_updated = P_pred - np.dot(np.dot(K, H), P_pred)
    return x_updated, P_updated, S

# T-Score Calculation
def calculate_t_score(y, y_pred, S):
    return np.dot(np.dot((y - y_pred).T, inv(S)), (y - y_pred))

# Dynamic Threshold Calculation
def dynamic_threshold(ship_data, confidence_factor=2):
    lat_std = ship_data["LAT"].std()
    lon_std = ship_data["LON"].std()
    position_threshold = np.sqrt(lat_std**2 + lon_std**2) * confidence_factor
    speed_threshold = ship_data["SOG"].std() * confidence_factor
    return position_threshold, speed_threshold

# Store all anomaly signals
all_signals = []

# Process each ship
grouped = data.groupby("MMSI")
total_ships = len(grouped)
ships_with_anomalies = []

for mmsi, ship_data in grouped:
    T_scores = []
    anomalies_detected = False

    first_row = ship_data.iloc[0]
    x_init = np.array([first_row["LAT"], first_row["LON"], first_row["SOG"], first_row["COG"]])
    P_init = np.eye(4) * 0.1

    position_threshold, speed_threshold = dynamic_threshold(ship_data)

    for i in range(1, len(ship_data)):
        current_data = ship_data.iloc[i]
        previous_data = ship_data.iloc[i - 1]
        time_diff = (current_data["BaseDateTime"] - previous_data["BaseDateTime"]).total_seconds()

        x_pred = state_transition_with_basis(x_init, previous_data["SOG"], previous_data["COG"],
                                             time_diff, basis_centers, weights)
        F = np.eye(4)
        Q = 0.01 * np.eye(4)
        x_pred, P_pred = kalman_filter_predict(x_init, P_init, F, Q)

        H = np.eye(4)
        R = 0.1 * np.eye(4)
        z = np.array([current_data["LAT"], current_data["LON"], current_data["SOG"], current_data["COG"]])
        x_updated, P_updated, S = kalman_filter_update(x_pred, P_pred, z, H, R)

        K = np.eye(2) * learning_rate
        weights = update_basis_weights(weights, K, z[:2], x_updated[:2])

        position_deviation = np.linalg.norm(z[:2] - x_updated[:2])
        speed_deviation = abs(z[2] - x_updated[2])

        all_signals.append({
            'MMSI': mmsi,
            'BaseDateTime': current_data['BaseDateTime'],
            'Position Deviation': position_deviation,
            'Speed Deviation': speed_deviation,
            'Position Threshold': position_threshold,
            'Speed Threshold': speed_threshold,
            'Anomaly': False
        })

        if position_deviation > position_threshold or speed_deviation > speed_threshold:
            anomalies_detected = True
            all_signals[-1]['Anomaly'] = True

        x_init = x_updated
        P_init = P_updated

    if anomalies_detected:
        ships_with_anomalies.append(mmsi)

# Save all signals
signals_df = pd.DataFrame(all_signals)

# Filter only anomalies
anomalies_df = signals_df[signals_df['Anomaly'] == True]

# Save anomalies to CSV
anomalies_df.to_csv(r"G:\SIH_FINAL_AIS_SATE\intergration\anomalies_detected_paper.csv", index=False)

print("Anomalies detected and saved to 'anomalies_detected_paper.csv'.")


# anomalies_df = signals_df[signals_df['Anomaly'] == True]
# for mmsi, anomaly_data in anomalies_df.groupby('MMSI'):
#     plt.figure(figsize=(10, 6))
#     plt.title(f"Anomalies Detected for MMSI: {mmsi}")
#     plt.scatter(anomaly_data['Timestamp'], anomaly_data['Position Deviation'], label="Position Deviation", color='red', s=10)
#     plt.scatter(anomaly_data['Timestamp'], anomaly_data['Speed Deviation'], label="Speed Deviation", color='blue', s=10)
#     plt.axhline(y=anomaly_data['Position Threshold'].iloc[0], color='red', linestyle='--', label='Position Threshold')
#     plt.axhline(y=anomaly_data['Speed Threshold'].iloc[0], color='blue', linestyle='--', label='Speed Threshold')
#     plt.xlabel("Timestamp")
#     plt.ylabel("Deviation")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# print("Anomaly detection completed. Results saved to 'anomalies.csv'.")