import cdsapi
import os
from datetime import date

# Zip files are extracted to data/era5grib/ by unzipERA5.py
output_dir = os.path.join("data", "ERA5_nordic", "raw")
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)

START_YEAR = 2022
END_YEAR = date.today().year  # ERA5 has ~5-day lag; CDS returns only available data
years = [str(y) for y in range(START_YEAR, END_YEAR + 1)]

# Time steps and area shared by both requests
# North, West, South, East: whole Norwegian coast (incl. Varangerfjord), Skagerrak, Kattegat and Danish North Sea
common = {
    "product_type": ["reanalysis"],
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "day": [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
    ],
    "time": ["00:00", "06:00", "12:00", "18:00"],
    "data_format": "grib",
    "download_format": "zip",
    "area": [80.75, 3, 54.5, 31.5]
}

client = cdsapi.Client(progress=True)

# Single level dataset (all years in one request)
client.retrieve("reanalysis-era5-single-levels", {
    **common,
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "total_precipitation",
        "land_sea_mask"
    ],
    "year": years
}).download("ERA5.zip")

# Pressure level dataset, one year per request to stay below the CDS size limit
for year in years:
    client.retrieve("reanalysis-era5-pressure-levels", {
        **common,
        "variable": [
            "geopotential",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity"
        ],
        "pressure_level": ["500", "700", "850", "925"],
        "year": [year]
    }).download(f"ERA5_pressure_{year}.zip")
