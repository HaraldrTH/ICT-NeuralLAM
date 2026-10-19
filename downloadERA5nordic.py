import cdsapi
import os


#Changing CWD to raw data folder, the zip files are extracted by unzipERA5.py
output_dir = os.path.join("data", "ERA5_nordic", "raw")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

os.chdir(output_dir)

#Time steps and area shared by both requests

years = [
    "2022", "2023",
    "2024", "2025", "2026"
]
common = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "mean_wave_direction",
        "mean_wave_period",
        "sea_surface_temperature",
        "significant_height_of_combined_wind_waves_and_swell",
        "surface_pressure",
        "total_precipitation",
        "peak_wave_period",
        "mean_zero_crossing_wave_period"
    ],
    "year": [
        "2022", "2023",
        "2024", "2025", "2026"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00",
        "06:00",
        "12:00",
        "18:00"
    ],
    "data_format": "grib",
    "download_format": "zip",
    #North, West, South, East: whole Norwegian coast (incl. Varangerfjord), Skagerrak, Kattegat and Danish North Sea
    "area": [80.75, 3, 54.5, 31.5]
}

client = cdsapi.Client(progress=True)

#Downloading single level dataset

single_levels = {
    **common,
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "mean_wave_direction",
        "mean_wave_period",
        "sea_surface_temperature",
        "significant_height_of_combined_wind_waves_and_swell",
        "total_precipitation",
        "land_sea_mask"
    ],
    "year": years
}
client.retrieve("reanalysis-era5-single-levels", single_levels).download("ERA5.zip")

#Downloading pressure level dataset, one year per request to stay below the CDS size limit

for year in years:
    pressure_levels = {
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
    }
    client.retrieve("reanalysis-era5-pressure-levels", pressure_levels).download(f"ERA5_pressure_{year}.zip")
