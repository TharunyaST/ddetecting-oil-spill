from extraction import extract_date , modify_path
from subnetprocessing import subnet_processing
from detection import detection
from locextrcation import location_extraction
from weatherapi import weather_api
from weatherdataexytraction import weather_data_extraction
from bbox import bbox
from datetime import datetime


global txtpath
txtpath="final_results.txt"

target_lat = 33.522823902460985
target_lon = -117.91402739036569

path=r"G:\SIH_FINAL_AIS_SATE\Final_satellite_intergration - Copy\extracted_SAR_data88ff\S1B_IW_GRDH_1SDV_20211003T014927_20211003T014952_028965_0374DA_3EE4.SAFE\measurement\s1b-iw-grd-vv-20211003t014927-20211003t014952-028965-0374da-001.tiff"
processed_image_path=subnet_processing(path)

annotation_path=modify_path(path)

processed_cropped_image_path=bbox(processed_image_path,annotation_path,target_lat,target_lon)

detection(processed_cropped_image_path)

day,month,year=extract_date(path)

lat1, lon1, lat2, lon2=location_extraction(processed_cropped_image_path,annotation_path)

file_path_grib=weather_api(int(year),int(month),int(day),float(lat1),float(lon1),float(lat2),float(lon2))

target_date=datetime(int(year),int(month),int(day))

weather_data_extraction(target_date,file_path_grib)