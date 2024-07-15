# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: tillmap
#     language: python
#     name: python3
# ---

# +
import os

# Directory where your files are stored
folder_path = "/Users/aminnorouzi/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/Double_Crop_Mapping/Joel_data/2021_2022"

# List all files in the folder
files = os.listdir(folder_path)

# Extract batch numbers from file names
batch_numbers = set()
for file in files:
    parts = file.split("_")
    if parts[-1].startswith("batch"):
        # Remove the file extension before converting to integer
        batch_number = int(parts[-1][5:].split(".")[0])
        batch_numbers.add(batch_number)

# Define the full set of batch numbers from 1 to 150
full_set = set(range(1, 151))

# Find missing batch numbers
missing_batches = full_set - batch_numbers

print("Missing batch numbers:", sorted(missing_batches))
