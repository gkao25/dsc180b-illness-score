import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import unary_union
import os
import re
import numpy as np
import glob
import time

predictor_cols = [
    'mean_wtd_moisture_1hr', 'mean_wtd_moisture_10hr', 
    'air_temperature_2m', 'air_relative_humidity_2m', 
    'wind_speed', 'accumulated_precipitation_amount',
    'surface_downwelling_shortwave_flux'
]

sd_gdf = gpd.read_file("SD_gjson.json")
sd_polygon = sd_gdf.unary_union

def convert_nc_to_csv(file_path):
    """
    Convert a .nc file to a CSV file with daily-averaged data filtered by a polygon.
    
    The CSV output is saved as:
        Cleaned_ENS/YYYY-MM/YYYY-MM-DD.csv

    Steps:
      1. Extract an 8-digit date (YYYYMMDD) from the input file path and format it as YYYY-MM-DD.
      2. Create a main output folder ("Cleaned_ENS") and a subfolder for the month (e.g., "2020-08").
      3. Read the NetCDF file and compute daily means.
      4. Convert the dataset to a DataFrame and create a "lat_lon" index.
      5. Read the polygon from "SD_gjson.json" and filter rows where the point is within the polygon.
      6. Write the filtered DataFrame to a CSV file.
    
    Args:
        file_path (str): Path to the .nc file.
    """
    match = re.search(r'(\d{8})', file_path)
    if not match:
        raise ValueError("No valid date found in the file_path.")
    date_str = match.group(1)             # e.g., "20200815"
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"  # "2020-08-15"
    
    # Create the main output folder and a subfolder (e.g., "2020-08")
    main_folder = "Cleaned_ENS"
    sub_folder = formatted_date[:7]
    output_folder = os.path.join(main_folder, sub_folder)
    
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the output file name
    output_file = os.path.join(output_folder, f"{formatted_date}.csv")
    
    # Open the dataset and compute daily means
    ds = xr.open_dataset(file_path)
    ds_daily = ds.resample(time="1D").mean()
    
    # Convert the dataset to a DataFrame and adjust the index
    beta = ds_daily.to_dataframe().reset_index()
    beta["lat_lon"] = list(zip(beta["latitude"], beta["longitude"]))
    beta = beta.set_index("lat_lon")
    beta = beta.drop(columns=["latitude", "longitude"])
    
    # Read the polygon from the GeoJSON file and filter the DataFrame
    beta["point"] = beta.index.map(lambda x: Point(x[1], x[0]))
    df = beta[beta["point"].apply(lambda pt: sd_polygon.contains(pt))].copy()
    df = df.drop(columns=["point"])

    # List of fire-related columns
    fire_cols = [
        'energy_release_component', 'ignition_component', 'fire_intensity_level',
        'forward_rate_of_spread', 'spread_component', 'burning_index', 'flame_length'
    ]

    for col in fire_cols:
        df[col + '_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df['fire_risk_composite'] = df[[col + '_norm' for col in fire_cols]].mean(axis=1)
    df['fire_risk_score'] = df['fire_risk_composite'] * 100
    df['wind_speed'] = np.sqrt(df['eastward_10m_wind']**2 + df['northward_10m_wind']**2)

    final_cols = predictor_cols + ['fire_risk_score']
    df = df[final_cols]
    
    # Write the filtered DataFrame to CSV
    df.to_csv(output_file)
    print(f"Filtered CSV written to: {output_file}")

if __name__ == "__main__":
    base_folder = "ens_gfs_001"
    
    # This pattern will match dfm*.nc files inside each subfolder of ens_gfs_001
    pattern = os.path.join(base_folder, "*", "dfm*.nc")
    
    nc_files = glob.glob(pattern)
    
    overall_start = time.time()
    
    for file_path in nc_files:
        start_time = time.time()
        convert_nc_to_csv(file_path)
        end_time = time.time()
        print(f"Time taken for {file_path}: {end_time - start_time:.2f} seconds")
    
    overall_end = time.time()
    print(f"Total time for converting all files: {overall_end - overall_start:.2f} seconds")