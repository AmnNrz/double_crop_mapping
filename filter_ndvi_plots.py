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

import pandas as pd
import numpy as np
import geopandas as gpd
import os
import re

# +
path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
                "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
                "Ph.D/Projects/Double_Crop_Mapping/")

plot_list = os.listdir(path_to_data + "Franklin_NDVI_plots/")

numbers = [re.search(r"_(\d+)\.pdf", file).group(1) for file in plot_list]
numbers = list(map(int, numbers))  # Convert the extracted numbers to integers
numbers
# shapefile = gpd.read_file(path_to_data + "GIS_files/WSDACrop_2023_WSUDoubleCrop.shp")
# shapefile.head(10)

# +
# # Filter shapefile for the fields that NDVI plots were downloaded
# filtered_shapefile = shapefile.loc[shapefile['OBJECTID'].isin(numbers)]
# filtered_shapefile.to_file(path_to_data + "ndvi_fields.shp")

# +
joel_selected_fields = gpd.read_file(
    path_to_data + "GIS_files/filter_Franklin_data/Joel_selected_fields.shp"
)

South_fields = gpd.read_file(
    path_to_data + "GIS_files/filter_Franklin_data/South_fields.shp"
)

North_fields = gpd.read_file(
    path_to_data + "GIS_files/filter_Franklin_data/North_fields.shp"
)

# +
import os
import shutil


source_folder = (path_to_data + "Franklin_NDVI_plots/")
destination_folder = (path_to_data + "Franklin_NDVI_plots/north_fields/")
subset_numbers = list(North_fields["OBJECTID"])

for filename in os.listdir(source_folder):
    if filename.endswith(".pdf"):
        # Extract the number from the filename
        try:
            file_number = int(filename.rstrip(".pdf").split("_")[-1])
        except ValueError:
            continue

        # Check if the number is in the subset
        if file_number in subset_numbers:
            source_file_path = os.path.join(source_folder, filename)
            destination_file_path = os.path.join(destination_folder, filename)
            shutil.copy(source_file_path, destination_file_path)
            print(f"Copied: {filename}")
