import rasterio
from datetime import datetime, timedelta
import csv
import numpy as np  # To handle data arrays for CSV writing
import os

def weather_data_extraction(target_date, file_path,save_dir_path):
    # print(file_path)
    # print(save_dir_path)
    # Initialize a list to store weather data records for CSV
    weather_records = []

    # Open the GRIB file
    with rasterio.open(file_path) as src:
        # Iterate over each band in the GRIB file
        for band_index in src.indexes:
            # Get the metadata tags for the band
            tags = src.tags(band_index)
            grib_element = tags.get("GRIB_ELEMENT", "No Element")  # The weather variable (e.g., U Wind, Pressure)
            grib_ref_time = tags.get("GRIB_REF_TIME", None)  # The reference time (start time of forecast)
            grib_forecast_seconds = tags.get("GRIB_FORECAST_SECONDS", None)  # The forecast time (in seconds)
            
            # Check if both reference time and forecast seconds are available
            if grib_ref_time and grib_forecast_seconds:
                # Convert GRIB reference time (Unix timestamp) to datetime
                try:
                    ref_time = datetime.utcfromtimestamp(int(grib_ref_time))  # Convert from Unix timestamp
                except ValueError:
                    print(f"Invalid GRIB_REF_TIME value: {grib_ref_time}")
                    continue

                # Calculate the forecast time by adding forecast seconds
                forecast_time = ref_time + timedelta(seconds=int(grib_forecast_seconds))
                
                # Check if the forecast time matches the target date and is at 23:00
                if forecast_time.date() == target_date.date() and forecast_time.hour == 23:
                    print(f"Data available for {grib_element} on {forecast_time.strftime('%Y-%m-%d %H:%M:%S')}")

                    # Read the data for the band
                    data = src.read(band_index)

                    # Flatten the data array to save it as a record
                    flattened_data = data.flatten()

                    # Append a record to the weather records list
                    weather_records.append({
                        "Forecast Time": forecast_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "Weather Element": grib_element,
                        "Data": flattened_data.tolist()  # Convert array to a list for saving
                    })
    output_csv_path = os.path.join(save_dir_path, "weather_data.csv")  # File path for saving the image
    print(output_csv_path)
    # Write the collected weather data to a CSV file
    with open(output_csv_path, mode='w', newline='') as csvfile:
        fieldnames = ["Forecast Time", "Weather Element", "Data"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in weather_records:
            writer.writerow({
                "Forecast Time": record["Forecast Time"],
                "Weather Element": record["Weather Element"],
                "Data": ";".join(map(str, record["Data"]))  # Join the data as a semicolon-separated string
            })

    return output_csv_path

    print(f"Weather data has been saved to {output_csv_path}")

