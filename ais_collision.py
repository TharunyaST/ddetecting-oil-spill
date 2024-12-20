import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
import time

# Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 1000  # Distance in meters

# Collision detection function
def detect_collisions_large(df_chunk, speed_threshold=3.0, bounding_box_distance=100):
    coords = df_chunk[['LAT', 'LON']].to_numpy()
    tree = cKDTree(coords)
    collision_details = []
    seen_pairs = set()  # To store unique ship pairs and their timestamp

    for i, ship in df_chunk.iterrows():
        lat1, lon1, speed1, time1 = ship['LAT'], ship['LON'], ship['SOG'], ship['BaseDateTime']
        if speed1 < speed_threshold:  # Skip slow-moving ships
            continue

        # Query nearby ships within the bounding box distance
        indices = tree.query_ball_point([lat1, lon1], bounding_box_distance / 111000)  # Convert meters to degrees
        indices = [j for j in indices if df_chunk.index[j] != i]  # Exclude the current ship itself

        for j in indices:
            other_ship = df_chunk.iloc[j]
            # Check if the timestamps match
            if ship['BaseDateTime'] == other_ship['BaseDateTime'] and ship['MMSI'] != other_ship['MMSI'] and other_ship['SOG'] > speed_threshold:
                distance = haversine(lat1, lon1, other_ship['LAT'], other_ship['LON'])
                if distance <= bounding_box_distance:
                    # Create a sorted key to avoid duplicate entries
                    pair_key = tuple(sorted([ship['MMSI'], other_ship['MMSI']])) + (time1,)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        collision_details.append({
                            'MMSI': ship['MMSI'],
                            'BaseDateTime': ship['BaseDateTime'],
                            'LAT': ship['LAT'],
                            'LON': ship['LON'],
                            'SOG': ship['SOG'],
                            'COG': ship['COG'],
                            'Heading': ship['Heading'],
                            'VesselName': ship['VesselName'],
                            'IMO': ship['IMO'],
                            'CallSign': ship['CallSign'],
                            'VesselType': ship['VesselType'],
                            'neighbor_MMSI': other_ship['MMSI'],
                            'neighbor_BaseDateTime': other_ship['BaseDateTime'],
                            'neighbor_ship_latitude': other_ship['LAT'],
                            'neighbor_ship_longitude': other_ship['LON'],
                            'neighbor_ship_SOG': other_ship['SOG'],
                            'neighbor_ship_COG': other_ship['COG'],
                            'neighbor_ship_Heading': other_ship['Heading'],
                            'neighbor_ship_name': other_ship['VesselName'],
                            'neighbor_ship_IMO': other_ship['IMO'],
                            'neighbor_ship_CallSign': other_ship['CallSign'],
                            'neighbor_ship_VesselType': other_ship['VesselType'],
                            'distance': distance
                        })
    return pd.DataFrame(collision_details)


# Wrapper function for multiprocessing
def process_chunk(chunk):
    return detect_collisions_large(chunk)

# Main script
if __name__ == '__main__':
    input_file = r'G:\SIH_FINAL_AIS_SATE\intergration\TEST.csv'
    output_file = r"G:\SIH_FINAL_AIS_SATE\intergration\collision.csv"
    chunk_size = 50000  # Adjust based on available memory
    num_processes = 4  # Match the number of CPU cores

    collision_df = pd.DataFrame()
    start_time = time.time()

    with Pool(processes=num_processes) as pool:
        for idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
            print(f"Processing chunk {idx + 1}...")
            results = pool.map(process_chunk, [chunk])  # Parallelize the processing
            for res in results:
                collision_df = pd.concat([collision_df, res], ignore_index=True)
            print(f"Chunk {idx + 1} processed.")

    # Deduplicate across chunks
    collision_df.drop_duplicates(subset=['MMSI', 'neighbor_MMSI', 'BaseDateTime'], inplace=True)

    # Save results
    collision_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
