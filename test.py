import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# User credentials
USERNAME = "lithish.it23@bitsathy.ac.in"
PASSWORD = "JS?Pxf6uuh_p*3_"

# Step 1: Generate an access token
def get_access_token(username, password):
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    payload = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password"
    }
    response = requests.post(url, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]

# Step 2: Query the OpenSearch Catalog
def search_products(access_token, lat, lon, start_date, end_date, product_type):
    base_url = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel1/search.json"
    params = {
        "startDate": start_date,
        "completionDate": end_date,
        "lon": lon,
        "lat": lat,
        "productType": product_type,
        "maxRecords": 10,  # Set max records to 10 or higher to get a list of products
        "sortParam": "startDate",
        "sortOrder": "ascending"
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()

# Step 3: Download the product
def download_product(access_token, product_id, file_path):
    base_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        with requests.get(base_url, headers=headers, stream=True) as response:
            if response.status_code == 401:
                print("Unauthorized error. Check token permissions or product access.")
                return None
            response.raise_for_status()
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        return None

# Main script to process the CSV and download products
def process_csv_and_download_data(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Ensure the directory exists
    SAVE_DIR = r"G:\SIH_FINAL_AIS_SATE\satellitedata"  # Specify your desired directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Generate access token
    access_token = get_access_token(USERNAME, PASSWORD)

    # Loop through the rows in the dataframe
    for index, row in df.iterrows():
        lat = row['LAT_x']
        lon = row['LON_x']
        base_datetime = row['BaseDateTime']
        
        # Parse BaseDateTime to get the starting date
        start_date = datetime.strptime(base_datetime, "%Y-%m-%d %H:%M:%S")
        end_date = start_date + timedelta(days=20)  # End date is 20 days from the start date

        # Convert dates to the required format for the API
        start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Search for products
        search_result = search_products(access_token, lat, lon, start_date_str, end_date_str, product_type="GRD")
        if "features" in search_result and search_result["features"]:
            product_id = search_result["features"][0]["id"]  # Get the first product found
            print(f"Product found. ID: {product_id}")

            # Generate a safe file name (replace ":" with "_")
            file_name = f"satellite_data_{start_date_str}_{end_date_str}.zip"
            file_name = file_name.replace(":", "_")  # Replace invalid characters
            file_path = os.path.join(SAVE_DIR, file_name)

            # Download the product
            downloaded_path = download_product(access_token, product_id, file_path)
            
            if downloaded_path:
                # Only update the CSV after the download is complete
                df.at[index, 'path'] = downloaded_path
            else:
                df.at[index, 'path'] = "Download Failed"
        else:
            df.at[index, 'path'] = "No product found"

    # Save the updated dataframe with the new 'path' column
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")

# Specify the input and output CSV files
input_csv = r'G:\SIH_FINAL_AIS_SATE\intergration\merged_output.csv'
output_csv = r'G:\SIH_FINAL_AIS_SATE\merged_output_with_paths.csv'

# Process the CSV and download satellite data
process_csv_and_download_data(input_csv, output_csv)
