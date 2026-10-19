import zipfile
import os

# Extracts zip files from data/ERA5_nordic/raw/ into data/era5grib/,
# which is the directory structure expected by prepare_era5.py
raw_dir = os.path.join("data", "ERA5_nordic", "raw")
grib_dir = os.path.join("data", "era5grib")

single_zip = os.path.join(raw_dir, "ERA5.zip")
if not os.path.exists(single_zip):
    raise FileNotFoundError(f"{single_zip} not found. Run downloadERA5nordic.py first.")

jobs = [(single_zip, os.path.join(grib_dir, "single_level"))]
for name in sorted(os.listdir(raw_dir)):
    if name.startswith("ERA5_pressure_") and name.endswith(".zip"):
        year = name[len("ERA5_pressure_"):-len(".zip")]
        jobs.append((os.path.join(raw_dir, name), os.path.join(grib_dir, "pressure", year)))

for zip_path, output_dir in jobs:
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for name in zip_file.namelist():
            print(f"Extracting {zip_path}: {name}")
        zip_file.extractall(output_dir)
    print(f"Done. Files extracted to {output_dir}")
