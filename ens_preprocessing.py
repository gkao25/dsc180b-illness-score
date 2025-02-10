import xarray as xr
from pathlib import Path

def clean_data(select_file):
    """
    This function accepts an input path that may be:
        - A single .nc file,
        - A directory containing multiple files,
        - Or a parent directory containing subfolders with files.
    
    Only files with names starting with "dfm" and ending with ".nc" are processed.
    
    input_path : str or Path
        Path to a single .nc file, a directory containing .nc files, or a parent directory with subfolders.
    
    Returns:
    dict
        A dictionary where keys are file paths (as strings) and values are processed xarray.Dataset objects.
    """
    
    input_path = Path(input_path)
    files_to_process = []
    
    # Determine whether input_path is a file or a directory
    if input_path.is_file():
        if input_path.name.startswith("dfm") and input_path.suffix == ".nc":
            files_to_process.append(input_path)
        else:
            print(f"File {input_path} does not match the required naming pattern ('dfm*.nc').")
    elif input_path.is_dir():
        files_to_process = list(input_path.rglob("dfm*.nc"))
    
    if not files_to_process:
        print("No matching .nc files found.")
        return {}
    
    processed_data = {}
    
    # List of fire-related variables to combine into a composite score
    fire_vars = [
        "energy_release_component",
        "ignition_component",
        "fire_intensity_level",
        "forward_rate_of_spread",
        "spread_component",
        "burning_index",
        "flame_length"
    ]
    
    for file in files_to_process:
        print(f"Processing file: {file}")
        
        # Open the dataset
        ds = xr.open_dataset(file)
        ds = ds.fillna(0)
        ds_daily = ds.resample(time="1D").mean()
        ds_daily["fire_risk"] = sum(ds_daily[var] for var in fire_vars)
        processed_data[str(file)] = ds_daily
    
    return processed_data