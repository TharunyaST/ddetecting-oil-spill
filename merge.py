import pandas as pd

# Read File 4
file4 = pd.read_csv(r"G:\SIH_FINAL_AIS_SATE\intergration\collision.csv")

# Map File 4 columns to their new names (adding `_x` suffix for main columns)
file4 = file4.rename(columns={
    'LAT': 'LAT_x', 
    'LON': 'LON_x', 
    'SOG': 'SOG_x', 
    'COG': 'COG_x', 
    'Heading': 'Heading_x',
    'VesselName': 'VesselName_x',
    'IMO': 'IMO_x', 
    'CallSign': 'CallSign_x',
    'VesselType': 'VesselType_x'
})

# Add missing columns to File 4 with 'NULL' values
all_columns = [
    'MMSI', 'BaseDateTime', 'LAT_x', 'LON_x', 'SOG_x', 'COG_x', 'Heading_x', 'VesselName_x',
    'IMO_x', 'CallSign_x', 'VesselType_x',
    'neighbor_MMSI', 'neighbor_BaseDateTime', 'neighbor_ship_latitude', 'neighbor_ship_longitude',
    'neighbor_ship_SOG', 'neighbor_ship_COG', 'neighbor_ship_Heading', 'neighbor_ship_name',
    'neighbor_ship_IMO', 'neighbor_ship_CallSign', 'neighbor_ship_VesselType', 'distance'
]

for col in all_columns:
    if col not in file4.columns:
        file4[col] = None  # Use None initially for missing columns

# Ensure column order matches the desired structure
file4 = file4[all_columns]

# Read other files
file1 = pd.read_csv(r"G:\SIH_FINAL_AIS_SATE\intergration\detected_anomalies_model.csv")
file2 = pd.read_csv(r"G:\SIH_FINAL_AIS_SATE\intergration\anomalies_detected_paper.csv")
file3 = pd.read_csv(r"G:\SIH_FINAL_AIS_SATE\intergration\path_anamoly.csv")

# Merge other files (existing logic)
df1 = file1[['MMSI', 'BaseDateTime']]
df2 = file2[['MMSI', 'BaseDateTime']]
df3 = file3[['MMSI', 'BaseDateTime']]

common_12 = pd.merge(df1, df2, on=['MMSI', 'BaseDateTime'])
common_13 = pd.merge(df1, df3, on=['MMSI', 'BaseDateTime'])
common_23 = pd.merge(df2, df3, on=['MMSI', 'BaseDateTime'])

all_common = pd.concat([common_12, common_13, common_23]).drop_duplicates()

merged_common_12 = pd.merge(all_common, file1, on=['MMSI', 'BaseDateTime'], how='left')
merged_common_13 = pd.merge(merged_common_12, file2, on=['MMSI', 'BaseDateTime'], how='left')
final_merged = pd.merge(merged_common_13, file3, on=['MMSI', 'BaseDateTime'], how='left')

# Remove duplicate columns
final_merged = final_merged.loc[:, ~final_merged.columns.str.contains('_y$')]

# Replace NaN values with 'NULL' explicitly
final_merged = final_merged.fillna('NULL')
file4 = file4.fillna('NULL')  # Replace None values in file4 with 'NULL'

# Concatenate File 4 data at the start of the merged DataFrame
final_result = pd.concat([file4, final_merged], ignore_index=True)

# Save final merged data
final_result.to_csv(r"G:\SIH_FINAL_AIS_SATE\intergration\merged_output.csv", index=False)
