import os
import zipfile
import pandas as pd
from extraction import extract_date, modify_path
from subnetprocessing import subnet_processing
from detection import detection
from locextrcation import location_extraction
from weatherapi import weather_api
from weatherdataexytraction import weather_data_extraction
from bbox import bbox
from datetime import datetime
from merge import merge1


# Load the CSV file
csv_path = r"G:\SIH_FINAL_AIS_SATE\merged_output_with_paths.csv"
data = pd.read_csv(csv_path)

if 'extracted_path' not in data.columns:
    data['extracted_path'] = None
    data.to_csv(csv_path, index=False)

i = 0
# Iterate over each row in the CSV
for index, row in data.iterrows():
    # Extract necessary values from the CSV row
    lat, lon = row["LAT_x"], row["LON_x"]
    zip_path = row["path"]  # Path to the zip file
    
    
    # Unzip the SAR data
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_path = os.path.join(os.path.dirname(zip_path), f"extracted_sar_data{i+1}")
        zip_ref.extractall(extract_path)
    i += 1
    # Navigate to the measurement folder
    for root, dirs, files in os.walk(extract_path):
        if root.endswith("measurement"):
            # Find the VV file
            vv_file = next((f for f in files if "vv" in f.lower() and f.endswith(".tiff")), None)
            if vv_file:
                vv_file_path = os.path.join(root, vv_file)
                break
    
    # Ensure we found the VV file
    if not vv_file:
        print(f"No VV file found in {zip_path}. Skipping...")
        continue
    print(lat,lon)
    # Start processing the VV file
    processed_image_path,save_dir_path = subnet_processing(vv_file_path)

    data.at[index, 'extracted_path'] = save_dir_path
    data.to_csv(csv_path, index=False)

    # print(processed_image_path)
    annotation_path = modify_path(vv_file_path)
    # print(annotation_path)
    processed_cropped_image_path = bbox(processed_image_path, annotation_path, lat, lon,save_dir_path )
    # print(processed_cropped_image_path)
    feature_path = detection(processed_cropped_image_path,save_dir_path)
    print(feature_path)

    # Extract the date
    day, month, year = extract_date(vv_file_path)

    # Extract location details
    lat1, lon1, lat2, lon2 = location_extraction(processed_cropped_image_path, annotation_path)

    # Fetch weather data
    file_path_grib = weather_api(
        int(year), int(month), int(day), 
        float(lat1), float(lon1), float(lat2), float(lon2),save_dir_path
    )

    # print(file_path_grib)

    # Extract weather data for the target date
    target_date = datetime(int(year), int(month), int(day))
    weather_path = weather_data_extraction(target_date, file_path_grib,save_dir_path)
    print(weather_path)

    merge1(feature_path,weather_path,save_dir_path)

    print(f"Processed SAR data from {zip_path} successfully.")

print("All SAR data processing completed.")
