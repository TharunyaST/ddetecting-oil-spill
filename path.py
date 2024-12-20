import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Define the GRU model class
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])  # Take output of last time step
        return out

# Load your trained model
input_size = 2  # Latitude and Longitude
hidden_size = 128
num_layers = 2
output_size = 2

# Initialize the model and load weights to CPU
model = GRUModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load(r"G:\SIH_FINAL_AIS_SATE\intergration\gru_model_epoch_5.pth", map_location=torch.device('cpu')))
model.eval()

# Load AIS data
df = pd.read_csv(r'G:\SIH_FINAL_AIS_SATE\intergration\dataset.csv')
df = df.iloc[:547466]

# Sort by MMSI and BaseDateTime
df.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)

# Normalize LAT and LON
scaler = MinMaxScaler()
df[['LAT', 'LON']] = scaler.fit_transform(df[['LAT', 'LON']])

# Parameters
sequence_length = 5  # Same sequence length used in training
distance_threshold = 0.1  # Threshold for anomaly detection

# Initialize storage for anomalies
anomalies_data = []

# Process ships
unique_mmsi = df['MMSI'].unique()
total_ships = len(unique_mmsi)

for mmsi in unique_mmsi:
    ship_data = df[df['MMSI'] == mmsi]
    lat_lon_data = ship_data[['LAT', 'LON']].values

    # Skip ships with insufficient data
    if len(lat_lon_data) <= sequence_length:
        continue

    # Prepare test data
    X_test = []
    y_actual = []

    for i in range(len(lat_lon_data) - sequence_length):
        X_test.append(lat_lon_data[i:i + sequence_length])
        y_actual.append(lat_lon_data[i + sequence_length])

    X_test = np.array(X_test)
    y_actual = np.array(y_actual)

    # Predict the path
    predicted_path = []
    anomalies = []

    with torch.no_grad():
        for idx, sequence in enumerate(X_test):
            # Move tensor to CPU
            input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # No .cuda() needed
            prediction = model(input_tensor).cpu().numpy()
            predicted_path.append(prediction[0])

            # Calculate Euclidean distance between actual and predicted positions
            distance = np.linalg.norm(y_actual[idx] - prediction[0])

            # Check if the distance exceeds the threshold
            if distance > distance_threshold:
                anomalies.append({
                    'MMSI': mmsi,
                    'BaseDateTime': ship_data.iloc[idx + sequence_length]['BaseDateTime'],
                    'Actual_LAT': y_actual[idx][0],
                    'Actual_LON': y_actual[idx][1],
                    'Predicted_LAT': prediction[0][0],
                    'Predicted_LON': prediction[0][1],
                    'Distance': distance,
                    **ship_data.iloc[idx + sequence_length].to_dict()
                })

    # Store anomalies for this ship
    anomalies_data.extend(anomalies)

# Save anomalies to a CSV file
anomalies_df = pd.DataFrame(anomalies_data)
anomalies_df.to_csv(r'G:\SIH_FINAL_AIS_SATE\intergration\path_anamoly.csv', index=False)

# Print summary
print(f"Total number of ships: {total_ships}")
print(f"Total number of anomalies detected: {len(anomalies_data)}")
print(f"Anomalies saved to path_anomalies.csv")
