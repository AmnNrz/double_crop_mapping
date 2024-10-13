# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="HE5ewNNo2qfK"
# This is getting worse by the day.
#
# Last time I used for-loop I was able to do 10 years, 500 fields at a time.
# Now, 1 year 20 fields gives trouble.
# (Dec 21, 2023)
#

# + colab={"base_uri": "https://localhost:8080/", "height": 90} executionInfo={"elapsed": 31047, "status": "ok", "timestamp": 1703740900855, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="4Gl6c3VSa89G" outputId="7d0c7e3d-22bf-46e7-d452-8e9386f59d30"
try:
  import shutup
except ImportError:
  # !pip install shutup
  import shutup

import warnings
warnings.filterwarnings('ignore')

shutup.please() # kill some of the messages

import pickle, time, datetime, scipy
import pandas as pd
import numpy as np
import geopandas as gpd
import json, geemap, ee, folium

import os, os.path, shutil, sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

# + [markdown] id="EqDUxjnG2_fX"
# #### Print Local Time for no reason!
#
# colab runs on cloud. So, the time is not our local time.
# This page is useful to determine how to do this.

# + colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 425, "status": "ok", "timestamp": 1703740901250, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="QpmbNbZs3AF3" outputId="cccc900f-ae11-4c4d-fafa-a7b5ea53498b"
# !rm /etc/localtime
# # !ln -s /usr/share/zoneinfo/US/Central /etc/localtime
# !ln -s /usr/share/zoneinfo/US/Pacific /etc/localtime
# !date

# + [markdown] id="4Hpg2fPA3DAy"
# ### geopandas and geemap must be installed every time.

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 248, "status": "ok", "timestamp": 1703740929865, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="PcLljbtK3FIR" outputId="7713770f-5674-4fea-eb07-d0df44122902"
# # !pip install geopandas geemap
# Installs geemap package
import subprocess

try:
  import geemap
except ImportError:
  print('geemap not installed. Must be installed every tim to run this notebook. Installing ...')
  subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

  print('geopandas not installed. Must be installed every time to run this notebook. Installing ...')
  subprocess.check_call(["python", '-m', 'pip', 'install', 'geopandas'])
  subprocess.check_call(["python", '-m', 'pip', 'install', 'google.colab'])

# + [markdown] id="P90coYYr3l-U"
# # **Authenticate and import libraries**
#
# We have to impor tthe libraries we need. Moreover, we need to Authenticate every single time!

# +
import ee

from google.auth import default
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/service-account-file.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/a.norouzikandelati/Google_stuff/gee_credentials/clear-shadow-332006-e8d8faf764f0.json"
# Obtain credentials with the appropriate scope
# Obtain credentials with additional scope for Google Drive
credentials, _ = default(scopes=[
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive'
])
# # Initialize the Earth Engine API with the specified project
# ee.Initialize(credentials=credentials, project='project-id')
ee.Initialize(credentials=credentials, project='clear-shadow-332006')
# -

# # Terminal arguments
# These are arguments passed to the script to run on Kamiak in parallel.

# block_count = 600
# batch_number = 1

# These are hard coded here based on the number of fields in the shapefile.
block_count = int(sys.argv[1])
batch_number = int(sys.argv[2])

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 28333, "status": "ok", "timestamp": 1703740961628, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="E81Q6rSR3Gai" outputId="aa226e34-ff90-47d7-f042-4757eb9894a2"
# try:
#   ee.Initialize()
# except Exception as e:
#   ee.Authenticate()
#   # this used to be empty. Now, it gives error.
#   # so, I put the project="ee-hnoorazarnasa"
#   # that is enabled as a google cloud project. GEE goes down the hill every day.
#   # ee.Initialize(project="ee-hnoorazarnasa")
#   ee.Initialize()

# + [markdown] id="3vStyGci31HH"
# ### **Mount Google Drive and import my Python modules**
#
# Here we are importing the Python functions that are written by me and are needed; ```NASA core``` and ```NASA plot core```.
#
# Note to self: These are on Google Drive now. Perhaps we can import them from GitHub.
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 28893, "status": "ok", "timestamp": 1703741003617, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="f_K_WFfu3rku" outputId="f9e878f4-2e52-47f7-9a61-30f770716249"
# Mount YOUR google drive in Colab
# from google.colab import drive
# drive.mount('/content/drive')
# import sys
# # sys.path.insert(0,"/content/drive/My Drive/Colab Notebooks/")
# sys.path.insert(0,"/content/drive/My Drive/WSU_job/joel_pipeline/")
import NASA_core as nc
import NASA_plot_core as ncp
import GEE_Python_core as gpc

# **Change Current directory to the Colab folder on Google Drive**
# import os
# os.chdir("/content/drive/My Drive/Colab Notebooks/") # Colab Notebooks
# # !ls

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 40, "status": "ok", "timestamp": 1703741003621, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="UFHkouIJ80i9" outputId="9520c02e-a408-422d-9453-e47ac0cd4acd"
data_base = "/home/a.norouzikandelati/Projects/Double_Crop_Mapping/"
shp_dir = data_base + "shapefiles/"
model_dir = data_base + "models/"
# -

# ### Please tell me where to look for the shapefile!
#
# <font size="5"><font color='red'>**Note:**</font></font> An ```ID``` column **must** be present in your shapefile. The code uses the ```ID``` to perform operations on each field. i.e. Each field (each row in data section of your shapefile) must be associated with a unique ID in a column called ```ID```.
#
# <font size="5"><font color='red'>**Note:**</font></font>
# Columns in the cell below must exist in your shapefile, otherwise, change
# them to the columns you have.
#
# <font size="5"><font color='red'>**Suggestion:**</font></font> Change column names in your shapefile, otherwise, you have to make more changes in the rest of the code.
#
# This is how columns can be renamed
#
# ```df.rename(columns={'old_col_1': 'new_col_1', 'old_col_2': 'new_col_2'}, inplace=True)```
#
# e.g.
#
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html

# + colab={"base_uri": "https://localhost:8080/", "height": 300} executionInfo={"elapsed": 2325, "status": "ok", "timestamp": 1703741009470, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="xDSoIeor4Dux" outputId="d61c1c75-4dea-4bd2-8716-4b42917851c9"
# %%time 
# we read our shapefile in to a geopandas data frame using the geopandas.read_file method
# we'll make sure it's initiated in the EPSG 4326 CRS
# SF_subdir, SF_file_name = "Grant_4Fields/", "Grant_4Fields"

# shapefile_name = "SF_2021_to_2023_DC_May272024.shp"
shapefile_name = "WSDACrop_2023_WSUDoubleCrop"
# path_to_shpfile = ("/home/a.norouzikandelati/Projects/Double_Crop_Mapping/"
#                     "GIS_files/qaqc_plots/")
path_to_shpfile = ("/home/a.norouzikandelati/Projects/Double_Crop_Mapping/"
                    "GIS_files/")
SF = gpd.read_file(path_to_shpfile + shapefile_name + ".shp", crs='EPSG:4326')
SF.head(2)


# shapefile_name = "GrantTest2020"
# path_to_shpfile = "/home/a.norouzikandelati/Projects/Double_Crop_Mapping/GrantTest2020/"
# SF = gpd.read_file(path_to_shpfile + shapefile_name + ".shp", crs='EPSG:4326')
# SF.head(2)

# +
# SF["DoubleCrop"].unique()
# SF = SF.loc[SF["DoubleCrop"] == "Yes"].copy()
# SF = SF.loc[SF["last_surve"] == 2023].copy()
# SF.shape
# -

# #  Column names **MUST** be consistent with the code
# Rename Columns of Joel to be consistent with what is in the code.
#
# What we need is
# ```keep_cols = ["ID", "Acres", "county", "CropTyp", "DataSrc", "Irrigtn", "LstSrvD", "geometry"]```

# <font size="4.5"><font color='red'>**Note:**</font></font>
#
# The following cell must be updated depending on the column names in the new shapefile created by Joel.

# +
SF.rename(columns={'OBJECTID': 'ID', 
                   'Irrigation': 'Irrigtn',
                   'County':'county',
                   'CropType':'CropTyp',
                   'DataSource':'DataSrc',
                   'LastSurvey':'LstSrvD'}, inplace=True)

SF.ID = SF.ID.astype(int)
SF.ID = SF.ID.astype(str)
keep_cols = ["ID", "Acres", "county", "CropTyp", "DataSrc", "Irrigtn", "LstSrvD", 'geometry']

SF= SF[keep_cols]
print (SF.shape)
# -

# ## Keep Eastern Washington and Irrigated fields

# +
SF.county = SF.county.str.replace(" ", "_")

# Eastern_counties = ["Grant", "Whitman", "Asotin",
#                     "Garfield", "Ferry", "Franklin", "Columbia", "Adams", "Benton" ,
#                     "Chelan", "Douglas", "Kittitas", "Klickitat", "Lincoln", "Okanogan", 
#                     "Spokane", "Stevens", "Yakima", 'Pend_Oreille', 'Walla_Walla']
Eastern_counties = ["Franklin"]

SF = SF[SF.county.isin(Eastern_counties)]
print (SF.shape)

# +
SF.Irrigtn = SF.Irrigtn.str.lower()
SF = nc.filter_out_nonIrrigated(SF)

SF.reset_index(drop=True, inplace=True)
print (f"{SF.shape = }")
SF.head(2)

SF = SF.loc[SF.Acres >= 10]
SF.shape
# -

# ## subset different batches for parallel running.

# +
field_count = len(SF.ID.unique())
print (f"{field_count = }")

block_size = np.floor(field_count/block_count)

start_row = int((batch_number-1) * block_size)
end_row = int(start_row + block_size - 1)

if batch_number<block_count:
    current_block = SF.iloc[start_row:end_row]
else:
    current_block = SF.iloc[start_row:]

SF = current_block.copy()
SF.reset_index(drop=True, inplace=True)
print (SF.shape)

# + [markdown] id="jUj6RFYnK1fW"
# <font size="4.5"><font color='red'>**Note:**</font></font>
#
# IF column ```ID``` is not there or is not populated do it here:

# + colab={"base_uri": "https://localhost:8080/", "height": 300} executionInfo={"elapsed": 229, "status": "ok", "timestamp": 1703741014012, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="8-VBWwyiJ2Yg" outputId="0a23f5ce-a628-4e14-def2-199d2b484ff2"
# I can only predict so many situations.

if not("ID" in SF.columns):
  SF["ID"] = ["field_" + str(x+1) + "_" + SF_file_name for x in SF.index]

if  type(SF.ID.values[1]) != str:
  SF["ID"] = ["field_" + str(x+1) + "_" + SF_file_name for x in SF.index]

SF.head(2)

# + [markdown] id="IJLulZzXIX9X"
# <font size="4.5"><font color='red'>**Note:**</font></font>
# If you want to susbset fields, this is the time. After this point the type of ```SF``` changes and doing some stuff becomes hard/impossible. You can make a back up of ```SF``` if you want. Like so ```SF_backup = SF.copy()```.

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 452, "status": "ok", "timestamp": 1703741026654, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="RpmpsYvPXMPN" outputId="13a68335-a132-4176-f49f-c7940b980ce4"
### subset of SF can be done like this:
# SF = SF[0:2] # replace 2 with the number you desire

# + colab={"base_uri": "https://localhost:8080/", "height": 203} executionInfo={"elapsed": 16, "status": "ok", "timestamp": 1703741026903, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="X9EGWmtxj7MO" outputId="e39a9e00-a5e1-44a4-bf9a-c132ba7a0ef0"
### for possible future use grab the data part of the shapefile

SF_data = SF[keep_cols].copy()
print (f"{SF_data.shape = }")

"""
   Drop extra useless columns. Saves space.**
   Also, GEE behaves strangely. It has problem with Notes column.
"""

# The only thing we need at this stage is only ID and geometry.
badCols = [x for x in list(SF.columns) if not (x in ["ID", "geometry"])]
SF = SF.drop(columns=badCols)
IDs = list(SF_data.ID.unique())

long_eq = "=============================================================================="
print (f"{type(SF) = }")
print (long_eq)
print (f"{SF.shape = }", )
print (long_eq)
SF.head(2)

# + [markdown] id="j35TrLzf5RDL"
# # **Form Geographical Regions**
#
#   - First, define a big region that covers Eastern Washington.
#   - Convert shapefile to ```ee.featurecollection.FeatureCollection```.
#
#
# ### Fetch data from GEE.
# <font size="4.5"><font color='red'>**NOTE:**</font></font>
# Try to make the following red box as small as possible. This can affect the resources CoLab let us use!

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 459, "status": "ok", "timestamp": 1703741047429, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="YLfgyhlf5Vor" outputId="ea810738-0eb6-410d-8450-fb6e68a19fe4"
xmin, xmax = -125.0, -116.0;
ymin, ymax = 45.0, 49.0;

xmed = (xmin + xmax) / 2.0;
ymed = (ymin+ymax) / 2.0;

WA1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed, ymax], [xmed, ymin], [xmin, ymin]]);
WA2 = ee.Geometry.Polygon([[xmed, ymin], [xmed, ymax], [xmax, ymax], [xmax, ymin], [xmed, ymin]]);
WA = [WA1,WA2];
# WA = [WA2];
big_rectangle = ee.FeatureCollection(WA);
if type(SF) != ee.featurecollection.FeatureCollection:
  SF = geemap.geopandas_to_ee(SF)

# + [markdown] id="kuwnMR-F5ckl"
# <font size="5"><font color='red'>**WARNING:**</font></font>
#
# For some reason the function ```feature2ee(.)``` does not work ***when*** it is imported from ```core``` module. (However, it works when it is directly written here!!!) So, What the happens with the rest of functions, e.g. smoothing functions, we want to use here?

# + [markdown] id="jEJMZfsh5gTg"
# # Visualize the big region encompassing the Eastern Washington

# + colab={"base_uri": "https://localhost:8080/", "height": 621, "referenced_widgets": ["981fa29c49ae4ccca3b6ddba753ba5be", "4c383bcd40334870b47202ad10fc8289", "9b87413e48eb461ea4c7fcc1ea72d35b", "2807d8fa311444c1ae3b1fa375fc09bf", "ca764c008f3b43fc85d1a637e5d592e2", "d95f05bb7e1e477395cc6629d0160572", "1fdd49d244e345888ca30acb3d699292", "9e1ba8cc40d74e1bbb6aa55cc79b58e1", "f5f67e930f4947e69f7cba030c9d336f", "e2f5bf3d825e493fae006bc775633f39", "7a111de8e8e94c4ba9849e56d456c12c", "cc2ed32b87ef4b5bb1cc4dd7022abf8d", "a90c616f3f6248d8833e26ef78283f87", "51bc239aa831477fab8f930c616df04c", "fd994c46c46442bfbd858f9b9ff969f0", "1e54745602c547888cc3380229104214", "6a54d4f117a04405971874fffc0854a9", "4dcc5d3720d94f599d9b2f554cf6a36e", "1565d1b19d2943be80bd685666c3f2b3", "a3b19cb434cc4d649be43cc52baec827", "f3b3d3fd2a3f4ef4a34b86eece9e9e5c", "68dfe65ce4db445e8e3a8870a5c90b2c", "fdf7a0279b834f49ad754bb73cf103ab", "065608173f334ea9b1615accc4615df6", "efdf57beff3c4285840846837c773b31", "550b66c31f074b42999f5e913a9eaf95", "058a9397f1f8496f8d039c8748a49da8"]} executionInfo={"elapsed": 1624, "status": "ok", "timestamp": 1703741055625, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="g7KhNNBX5lvc" outputId="c28ec42f-f1ef-4acb-a400-23c4565cafb9"
Map = geemap.Map(center=[ymed, xmed], zoom=7)
Map.addLayer(WA1, {'color': 'red'}, 'Western Half')
Map.addLayer(WA2, {'color': 'red'}, 'Eastern Half')
Map.addLayer(SF, {'color': 'blue'}, 'Fields')
Map

# + [markdown] id="RGCC6axi5nMl"
# ### Define Parameters

# + colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 240, "status": "ok", "timestamp": 1703741123099, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="YJp7YEu35r5R" outputId="e6266373-1182-4186-a5fe-945455f92feb"
# Date fromat for EE YYYY-MM-DD
shape_file_year = 2024
start_date = str(shape_file_year) + "-01-01"
end_date =   str(shape_file_year+1) + "-08-01"
print ("we are looking at {} to {}.".format(start_date, end_date))
# sentinel parameter for removing cloudy pixels in sentinel images.
# Change at your own risk.
cloud_perc = 70

# + [markdown] id="8nPIerfp5tQd"
# ### Fetch data from GEE.
# <font size="4.5"><font color='red'>**NOTE:**</font></font>
# We are using Sentinel here. More data. If people are obliged to using Landsat the code needs to change!

# + colab={"base_uri": "https://localhost:8080/", "height": 713} executionInfo={"elapsed": 311414, "status": "ok", "timestamp": 1703743123831, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="npYju8WxjDF6" outputId="5c27c88b-3fcf-4457-ecf9-f4102a6619c7"
# %%time
imageC = gpc.extract_sentinel_IC(big_rectangle, start_date, end_date, cloud_perc);
# print ("The size of image collection is [{:.0f}].".format(imageC.size().getInfo()))

reduced = gpc.mosaic_and_reduce_IC_mean(imageC, SF, start_date, end_date)
# print (type(reduced))
# print ("The size of reduced is [{:.0f}].".format(reduced.size().getInfo()))

needed_columns = ["ID", "EVI", 'NDVI', "system_start_time"]
# -

# ## Export output to Google Drive
#
# <font size="4.5"><font color='red'>**Suggestion:**</font></font>
# We advise you to Export the data. If Python/CoLab kernel dies, then,
# previous steps should not be repeated.
#
# <font size="4.5"><font color='red'>**Suggestion:**</font></font>
# Install GEE packages on your local machine and run it locally. CoLab WILL die.

# # Delete all files from service account drive

# +
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Path to your service account key file
SERVICE_ACCOUNT_FILE = '/home/a.norouzikandelati/Google_stuff/gee_credentials/clear-shadow-332006-e8d8faf764f0.json'

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

# Authenticate and construct service
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

# Function to delete all files
def delete_all_files():
    results = service.files().list(pageSize=100, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    if not items:
        print('No files found.')
    else:
        for item in items:
            try:
                service.files().delete(fileId=item['id']).execute()
                print(f"Deleted file: {item['name']} (ID: {item['id']})")
            except Exception as e:
                print(f"Failed to delete {item['name']} (ID: {item['id']}): {e}")

# Call the function to delete all files
delete_all_files()
# -

# # Export data to Google Drive and read it again

# +
# # %%time

# import io
# import os
# import pickle
# import time
# import pandas as pd
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from google.oauth2 import service_account

# # Function to authenticate google drive and create a drive service
# def authenticate_Gdrive():
#     SCOPES = ['https://www.googleapis.com/auth/drive']
#     token_pickle = '/home/a.norouzikandelati/Google_stuff/Gdrive_credentials/token.pkl'
#     credentials_json = '/home/a.norouzikandelati/Google_stuff/Gdrive_credentials/credentials.json'
    
#     creds = None
#     # Load credentials from file if they exist
#     if os.path.exists(token_pickle):
#         with open(token_pickle, 'rb') as token:
#             creds = pickle.load(token)
#     # Check if the credentials are valid
#     if not creds or not creds.valid:
#         if creds and creds.expired:
#             try:
#                 creds.refresh(Request())
#             except Exception as e:
#                 print("Failed to refresh the access token. Re-authenticating...")
#                 flow = InstalledAppFlow.from_client_secrets_file(credentials_json, SCOPES)
#                 creds = flow.run_local_server(port=0)
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(credentials_json, SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open(token_pickle, 'wb') as token:
#             pickle.dump(creds, token)
#     service = build('drive', 'v3', credentials=creds)
#     return service

# # Function to authenticate service account
# def authenticate_service_account():
#     # Path to your service account key file
#     SERVICE_ACCOUNT_FILE = '/home/a.norouzikandelati/Google_stuff/gee_credentials/clear-shadow-332006-e8d8faf764f0.json'

#     # Define the scopes
#     SCOPES = ['https://www.googleapis.com/auth/drive']

#     # Authenticate and construct service
#     credentials = service_account.Credentials.from_service_account_file(
#         SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#     service = build('drive', 'v3', credentials=credentials)
#     return service

# # Function to download a file from Google Drive
# def download_file(service, file_id, file_path):
#     request = service.files().get_media(fileId=file_id)
#     fh = io.BytesIO()
#     downloader = MediaIoBaseDownload(fh, request)
#     done = False
#     while not done:
#         status, done = downloader.next_chunk()
#     fh.seek(0)
#     with open(file_path, 'wb') as f:
#         f.write(fh.read())
#     print('Download Complete')

# # Function to find the file on Google Drive
# def find_file(service, file_name):
#     response = service.files().list(q=f"name='{file_name}'", spaces='drive', fields="nextPageToken, files(id, name)").execute()
#     for file in response.get('files', []):
#         return file.get('id')
#     return None

# # Function to delete a specific file by its ID
# def delete_file(file_id):
#     # Path to your service account key file
#     SERVICE_ACCOUNT_FILE = '/home/a.norouzikandelati/Google_stuff/gee_credentials/clear-shadow-332006-e8d8faf764f0.json'

#     # Define the scopes
#     SCOPES = ['https://www.googleapis.com/auth/drive']

#     # Authenticate and construct service
#     credentials = service_account.Credentials.from_service_account_file(
#         SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#     service = build('drive', 'v3', credentials=credentials)

#     # Delete the specific file
#     try:
#         service.files().delete(fileId=file_id).execute()
#         print(f"Deleted file with ID: {file_id}")
#     except Exception as e:
#         print(f"Failed to delete file (ID: {file_id}): {e}")
        
# # Function to initiate and monitor the export task
# def export_data_to_drive(collection, description, folder, file_prefix, file_format, selectors):
#     task = ee.batch.Export.table.toDrive(
#         collection=collection.select(selectors),
#         description=description,
#         folder=folder,
#         fileNamePrefix=file_prefix,
#         fileFormat=file_format
#     )
#     task.start()

#     while task.active():
#         print('Polling for task (id: {}).'.format(task.id))
#         time.sleep(30)

#     if task.status()['state'] == 'COMPLETED':
#         print('Export completed successfully!')
#         service = authenticate_Gdrive()
#         file_name = f"{file_prefix}.csv"
#         file_id = find_file(service, file_name)
#         if file_id:
#             download_file(service, file_id, file_name)
#             df = pd.read_csv(file_name)
#             print(df.head())
#         else:
#             print('File not found.')
#         # service = authenticate_service_account()
#         # file_name = f"{file_prefix}.csv"
#         # file_id = find_file(service, file_name)
#         # delete_file(file_id)
#     else:
#         print('Error with export:', task.status())
#     return df

# # Example usage of the function
# selected_properties = ["ID", "EVI", 'NDVI', "system_start_time"]
# file_name = shapefile_name + "_batch" + str(batch_number)
# df = export_data_to_drive(reduced, str(batch_number), 'doubleCropping_data', file_name, 'CSV', selected_properties)
# df.head(2)


# +
# %%time

import io
import os
import pickle
import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# Function to authenticate and create a drive service
from google.oauth2 import service_account
from googleapiclient.discovery import build

def authenticate_drive():
    # Path to your service account key file
    SERVICE_ACCOUNT_FILE = '/home/a.norouzikandelati/Google_stuff/gee_credentials/clear-shadow-332006-e8d8faf764f0.json'

    # Define the scopes
    SCOPES = ['https://www.googleapis.com/auth/drive']

    # Authenticate and construct service
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

# Replace all instances where you need an authenticated drive service with authenticate_drive()


# Function to download a file from Google Drive
def download_file(service, file_id, file_path):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    with open(file_path, 'wb') as f:
        f.write(fh.read())
    print('Download Complete')

# Function to find the file on Google Drive
def find_file(service, file_name):
    response = service.files().list(q=f"name='{file_name}'", spaces='drive', fields="nextPageToken, files(id, name)").execute()
    for file in response.get('files', []):
        return file.get('id')
    return None

# Function to delete a specific file by its ID
def delete_file(file_id):
    # Path to your service account key file
    SERVICE_ACCOUNT_FILE = '/home/a.norouzikandelati/Google_stuff/gee_credentials/clear-shadow-332006-e8d8faf764f0.json'

    # Define the scopes
    SCOPES = ['https://www.googleapis.com/auth/drive']

    # Authenticate and construct service
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)

    # Delete the specific file
    try:
        service.files().delete(fileId=file_id).execute()
        print(f"Deleted file with ID: {file_id}")
    except Exception as e:
        print(f"Failed to delete file (ID: {file_id}): {e}")
        
# Function to initiate and monitor the export task
def export_data_to_drive(collection, description, folder, file_prefix, file_format, selectors):
    task = ee.batch.Export.table.toDrive(
        collection=collection.select(selectors),
        description=description,
        folder=folder,
        fileNamePrefix=file_prefix,
        fileFormat=file_format
    )
    task.start()

    while task.active():
        print('Polling for task (id: {}).'.format(task.id))
        time.sleep(30)

    if task.status()['state'] == 'COMPLETED':
        print('Export completed successfully!')
        service = authenticate_drive()
        file_name = f"{file_prefix}.csv"
        file_id = find_file(service, file_name)
        if file_id:
            download_file(service, file_id, file_name)
            df = pd.read_csv(file_name)
            delete_file(file_id)
            print(df.head())
        else:
            print('File not found.')
    else:
        print('Error with export:', task.status())
    return df

# Example usage of the function
selected_properties = ["ID", "EVI", 'NDVI', "system_start_time"]
file_name = shapefile_name + "_batch" + str(batch_number)
df = export_data_to_drive(reduced, str(batch_number), 'doubleCropping_data', file_name, 'CSV', selected_properties)
df.head(2)

# + [markdown] id="TXlgCZM27JLQ"
# # **Smooth the data**
#
# This is the end of Earh Engine Part. Below we start smoothing the data and carry on!
#
# First, all these steps can be done behind the scene. But doing them here, one at a time, has the advantage that if something goes wrong in the middle, then
# we do not lose the good stuff that was done earlier!
# For example, of one of the Python libraries/packages needs to be updated in the middle of the way
# we do not have to start doing everything from the beginning!
# <p>&nbsp;</p>
#
# Start with converting the type of ```reduced``` from ```ee.FeatureCollection``` to ```dataframe```.
#
# - For some reason when converting the ```ee.FeatureCollection``` to ```dataframe``` the function has a problem with the ```Notes``` column! So, I remove the unnecessary columns.
#
# **NA removal**
#
# Even though logically and intuitively all the bands should be either available or ```NA```, I have seen before that sometimes ```EVI``` is NA while ```NDVI``` is not. Therefore, I had to choose which VI we want to use so that we can clean the data properly. However, I did not see that here.  when I was testing this code for 4 fields.
#
# Another suprising observation was that the output of Colab had more data compared to its JS counterpart!!!
#
# ### **Define the VI parameter we want to work with**

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1702075946373, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="KXFjfXwg7Nx-" outputId="cf9e2fe6-9444-4b43-8fdf-892d0c8173cf"
VI_idx = "NDVI"
# -

reduced = df.copy()

# + colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 278, "status": "ok", "timestamp": 1702075946648, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="vkE2_Q857Zqy" outputId="e4f3bea0-6038-4eb8-a07d-3465e62a77a2"
## drop the NAs in the given VI:
# reduced = reduced[reduced["system_start_time"].notna()]
reduced = reduced[reduced[VI_idx].notna()]
reduced.reset_index(drop=True, inplace=True)

# Add human readable time to the dataframe
reduced = nc.add_human_start_time_by_system_start_time(reduced)
reduced.drop(columns=["system:index", ".geo"], inplace=True)
reduced.head(2)
# -

reduced = reduced.loc[reduced["human_system_start_time"] <= "2024-08-01"]
# reduced["human_system_start_time"][0]
max(reduced["human_system_start_time"])

# + colab={"base_uri": "https://localhost:8080/", "height": 291} executionInfo={"elapsed": 581, "status": "ok", "timestamp": 1702075947227, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="kpza5dqs7kec" outputId="39c14c85-b262-4457-92bb-545458d60b28"
#  Pick a field
a_field = reduced[reduced.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(axis='y', which="both")

ax.scatter(a_field['human_system_start_time'], a_field[VI_idx], s=40, c='#d62728');
ax.plot(a_field['human_system_start_time'], a_field[VI_idx],
        linestyle='-',  linewidth=3.5, color="#d62728", alpha=0.8,
        label=f"raw {VI_idx}")
plt.ylim([-0.5, 1.2]);
ax.legend(loc="lower right");
# ax.set_title(a_field.CropTyp.unique()[0]);

# + [markdown] id="oXLzD52E7r6F"
# **Remove outliers**

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1702075947227, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="W0PYI27P7wkO" outputId="2dd952b1-312c-49b5-da11-413d27703f1f"
reduced["ID"] = reduced["ID"].astype(str)
# p = np.sort(reduced["ID"].unique())
# -

reduced = reduced[['EVI', 'ID', 'NDVI', 'system_start_time', 'human_system_start_time']]

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 293, "status": "ok", "timestamp": 1702075947517, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="K_sLBC0E72uQ" outputId="9007d0ae-9fc7-410d-e4fd-cbf9622d1357"
no_outlier_df = pd.DataFrame(data = None,
                             index = np.arange(reduced.shape[0]),
                             columns = reduced.columns)
counter = 0
row_pointer = 0
for a_poly in reduced["ID"].unique():
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = reduced[reduced["ID"]==a_poly].copy()
    # small fields may have nothing in them!
    if curr_field.shape[0] > 2:
        ##************************************************
        #
        #    Set negative index values to zero.
        #
        ##************************************************
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = VI_idx)
        no_Outlier_TS.loc[no_Outlier_TS[VI_idx
                                        ] < 0 , VI_idx] = 0

        """
        it is possible that for a field we only have x=2 data points
        where all the EVI/NDVI is outlier. Then, there is nothing to
        use for interpolation. So, hopefully interpolate_outliers_EVI_NDVI is returning an empty data table.
        """
        if len(no_Outlier_TS) > 0:
            no_outlier_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
            counter += 1
            row_pointer += curr_field.shape[0]

# Sanity check. Will neved occur. At least should not!
no_outlier_df.drop_duplicates(inplace=True)

# + [markdown] id="ICwyn6pT73NO"
# **Remove the jumps**
#
# Maybe we can remove old/previous dataframes to free memory up!

# + colab={"base_uri": "https://localhost:8080/", "height": 69} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1702075947517, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="cpQrzfGq8BPo" outputId="97dee443-851b-44da-a865-f86163430b65"
noJump_df = pd.DataFrame(data = None,
                         index = np.arange(no_outlier_df.shape[0]),
                         columns = no_outlier_df.columns)
counter, row_pointer = 0, 0

for a_poly in no_outlier_df["ID"].unique():
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = no_outlier_df[no_outlier_df["ID"]==a_poly].copy()

    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)

    ################################################################

    no_Outlier_TS = nc.correct_big_jumps_1DaySeries_JFD(dataTMS_jumpie = curr_field,
                                                        give_col = VI_idx
                                                        ,
                                                        maxjump_perDay = 0.018)

    noJump_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]

noJump_df['human_system_start_time'] = pd.to_datetime(noJump_df['human_system_start_time'])

# Sanity check. Will neved occur. At least should not!
print ("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape))
noJump_df.drop_duplicates(inplace=True)
print ("Shape of noJump_df after dropping duplicates is {}.".format(noJump_df.shape))

del(no_Outlier_TS)

# + [markdown] id="vI3Lzuvx8FLI"
# **Regularize**
#
# Here we regularize the data. "Regularization" means pick a value for every 10-days. Doing this ensures
#
# 1.   all inputs have the same length,
# 2.   by picking maximum value of a VI we are reducing the noise in the time-series by eliminating noisy data points. For example, snow or shaddow can lead to understimating the true VI.
#
# Moreover, here, I am keeping only 3 columns. As long as we have ```ID``` we can
# merge the big dataframe with the final result later, here or externally.
# This will reduce amount of memory needed. Perhaps I should do this
# right the beginning.

# + colab={"base_uri": "https://localhost:8080/", "height": 139} executionInfo={"elapsed": 1577, "status": "ok", "timestamp": 1702075949091, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="QUDxdJ948IaK" outputId="34ead020-e32e-415f-8a83-dc59d7587d51"
# %%time

# define parameters
regular_window_size = 10
reg_cols = ['ID', 'human_system_start_time', VI_idx] # system_start_time list(noJump_df.columns)

st_yr = noJump_df.human_system_start_time.dt.year.min()
end_yr = noJump_df.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366 # 14 years, each year 366 days!

no_steps = int(np.ceil(no_days / regular_window_size)) # no_days // regular_window_size

nrows = no_steps * len(IDs)
print('st_yr is {}.'.format(st_yr))
print('end_yr is {}.'.format(end_yr))
print('nrows is {}.'.format(nrows))
print (long_eq)


regular_df = pd.DataFrame(data = None,
                         index = np.arange(nrows),
                         columns = reg_cols)
counter, row_pointer = 0, 0

for a_poly in noJump_df["ID"].unique():
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = noJump_df[noJump_df["ID"]==a_poly].copy()
    ################################################################
    # Sort by date (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)

    ################################################################
    regularized_TS = nc.regularize_a_field(a_df = curr_field, \
                                           V_idks = VI_idx, \
                                           interval_size = regular_window_size,\
                                           start_year = st_yr, \
                                           end_year = end_yr)

    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = VI_idx)
    # if (counter == 0):
    #     print ("regular_df columns:",     regular_df.columns)
    #     print ("regularized_TS.columns", regularized_TS.columns)

    ################################################################
    # row_pointer = no_steps * counter

    """
       The reason for the following line is that we assume all years are 366 days!
       so, the actual thing might be smaller!
    """
    # why this should not work?: It may leave some empty rows in regular_df
    # but we drop them at the end.
    regular_df[row_pointer : (row_pointer+regularized_TS.shape[0])] = regularized_TS.values
    row_pointer += regularized_TS.shape[0]
    counter += 1

regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
regular_df.drop_duplicates(inplace=True)
regular_df.dropna(inplace=True)

# Sanity Check
regular_df.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
regular_df.reset_index(drop=True, inplace=True)

del(noJump_df)
# -

regular_df = regular_df.loc[regular_df["human_system_start_time"] <= "2024-08-01"]

# + colab={"base_uri": "https://localhost:8080/", "height": 291} executionInfo={"elapsed": 272, "status": "ok", "timestamp": 1702075949360, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="eNypD0y98LFQ" outputId="c48609c6-2b4c-4dc6-8999-2c002ed51201"
#  Pick a field
a_field = regular_df[regular_df.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3), sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'],
        a_field[VI_idx],
        linestyle='-', label=VI_idx, linewidth=3.5, color="dodgerblue", alpha=0.8)

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);
# -

a_field['human_system_start_time']

# + [markdown] id="VO4WOlWU8OgG"
# **Savitzky-Golay Smoothing**

# + colab={"base_uri": "https://localhost:8080/", "height": 69} executionInfo={"elapsed": 206, "status": "ok", "timestamp": 1702075949564, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="mfdW8BFk8TgT" outputId="82da8d9b-c347-44dc-a8ba-d300230c7235"
# %%time
counter = 0
window_len, polynomial_order = 7, 3

for a_poly in regular_df["ID"].unique():
    if (counter % 300 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = regular_df[regular_df["ID"]==a_poly].copy()

    # Smoothen by Savitzky-Golay
    SG = scipy.signal.savgol_filter(curr_field[VI_idx].values, window_length=window_len, polyorder=polynomial_order)
    SG[SG > 1 ] = 1 # SG might violate the boundaries. clip them:
    SG[SG < -1 ] = -1
    regular_df.loc[curr_field.index, VI_idx] = SG
    counter += 1

# + colab={"base_uri": "https://localhost:8080/", "height": 291} executionInfo={"elapsed": 669, "status": "ok", "timestamp": 1702075950230, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="-wFLUTe78Ty5" outputId="e58e7e31-f35f-4930-b4f5-819b5f4c7d22"
# Pick a field
an_ID = IDs[0]
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3), sharex='col', sharey='row', gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], a_field[VI_idx],
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8, label=f"smooth {VI_idx}")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);
# -

a_field

# + colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1702075950230, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="fBI3nMmo8Vby" outputId="66a7580a-9075-4ebb-c4ee-9e7c0d1b3f6c"
regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
# regular_df = pd.merge(regular_df, SF_data, on=['ID'], how='left') # we can do this later.
regular_df.reset_index(drop=True, inplace=True)
regular_df = nc.initial_clean(df=regular_df, column_to_be_cleaned = VI_idx)
regular_df.head(2)

# +
import pickle
from datetime import datetime

filename = path_to_shpfile + "qaqc_NDVI_TS.sav"

export_ = {"SG_TS": regular_df, 
           "source_code" : "pipeline",
           "Author": "ANK",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))
# -

regular_df.head(2)

# + [markdown] id="V9tQNxIZ8XTK"
# **Widen the data to use with ML (other than DL)**
#
# <font size="4.5"><font color='red'>**Note:**</font></font>
# At this point we have not released any model other than ```DL``` model.

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1702075950230, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="YyQuPZZ_8i1M" outputId="4ad1f616-8cca-433d-e4fa-a31bbafb6720"
model = "DL"

if model != "DL":
    VI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37)]
    columnNames = ["ID", "year"] + VI_colnames

    years = regular_df.human_system_start_time.dt.year.unique()
    no_rows = len(IDs) * len(years)

    data_wide = pd.DataFrame(columns=columnNames, index=range(no_rows))
    data_wide.ID = list(IDs) * len(years)
    data_wide.sort_values(by=["ID"], inplace=True)
    data_wide.reset_index(drop=True, inplace=True)
    data_wide.year = list(years) * len(IDs)

    for an_ID in regular_df.ID.unique():
        curr_field = regular_df[regular_df.ID == an_ID]
        curr_years = curr_field.human_system_start_time.dt.year.unique()
        for a_year in curr_years:
            curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]
            data_wide_indx = data_wide[(data_wide.ID == an_ID) & (data_wide.year == a_year)].index
            if VI_idx == "EVI":
                data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = curr_field_year.EVI.values[:36]
            elif VI_idx == "NDVI":
                data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = curr_field_year.NDVI.values[:36]

# + [markdown] id="aiCPCDzt8kNc"
# ### Please tell me where to look for the trained models and I will make you happy!
#
# ```model_dir``` is defined on top of the notebook. But, I repeat it here!

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 20664, "status": "ok", "timestamp": 1702075970884, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="XUZ8ca9y9Txy" outputId="50a16398-0c92-4be2-9c81-b05354387512"
if model == "KNN":
    winnerModel = "KNN_SG_NDVI_Oct17_AccScoring_Oversample_SR3.sav"
elif model == "SVM":
    winnerModel = "SG_NDVI_SVM_NoneWeight_00_Oct17_AccScoring_Oversample_SR3.sav"
elif model =="RF":
    winnerModel = "SG_NDVI_RF_grid_2_Oct17_AccScoring_Oversample_SR5.sav"
else:
    winnerModel = "01_TL_NDVI_SG_train80_Oct17_oversample5.h5"
    winnerModel = "01_TL_NDVI_SG_train80_Oct17.h5"

###
### Load model and predict
###
if winnerModel.endswith(".sav"):
    # ML_model = pickle.load(open(model_dir + winnerModel, "rb"))
    ML_model = pd.read_pickle(model_dir + winnerModel)
    predictions = ML_model.predict(data_wide.iloc[:, 2:])
    pred_colName = model + "_" + VI_idx  + "_preds"
    A = pd.DataFrame(columns=["ID", "year", pred_colName])
    A.ID = data_wide.ID.values
    A.year = data_wide.year.values
    A[pred_colName] = predictions
    predictions = A.copy()
    del A
else:
    from tensorflow.keras.utils import to_categorical, load_img, img_to_array
    from keras.models import Sequential, Model, load_model
    from keras.applications.vgg16 import VGG16
    import tensorflow as tf

    # from keras.optimizers import SGD
    from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.optimizers import SGD
    # from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    #### Form predictions' dataframe
    predictions = pd.DataFrame({"ID": list(regular_df.ID.unique())})
    predictions["prob_single"] = -1.0

    ML_model = load_model(model_dir + winnerModel) # load model

    # image_dir = '/content/drive/MyDrive/colab_outputs/'
    image_dir = data_base + "joel_figures/"
    image_name = image_dir + "fly_test.jpg"
    for an_ID in regular_df.ID.unique():
        crr_fld = regular_df[regular_df.ID == an_ID]
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 2.5)
        ax.grid(False)
        ax.plot(
            crr_fld["human_system_start_time"], crr_fld[VI_idx], c="dodgerblue", linewidth=5
        )
        ax.axis("off")
        left = crr_fld["human_system_start_time"].values[0]
        right = crr_fld["human_system_start_time"].values[-1]
        ax.set_xlim([left, right]);
        # the following line also works
        ax.set_ylim([-0.005, 1]);

        plt.savefig(fname = image_name, dpi = 200, bbox_inches = "tight", facecolor = "w")
        plt.close("all")

        img = nc.load_image(image_name)
        predictions.loc[predictions.ID == an_ID, "prob_single"] = ML_model.predict(img, verbose=False)[0][0]
# -

model_dir

SF_data.head(2)

# + colab={"base_uri": "https://localhost:8080/", "height": 864} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1702075970885, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="KkIPBr719w_h" outputId="3e0ef27e-1cbb-4ea5-dfc7-5657e22370c5"
predictions = pd.merge(predictions, SF_data, on=['ID'], how='left')
predictions.sort_values(by=["ID"], inplace=True)
predictions.drop(columns=["geometry"], inplace=True)
predictions.head(5)

# + [markdown] id="o4o58W0Tbqc4"
# <font size="5"><font color='red'>**Note:**</font></font>
# The cut-off threshold for this model is $p=0.3$. If model changes this needs to be adjusted as well.
# -

predictions[predictions["ID"] == "18523"]

# + colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1702075970885, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="VceC3NbKbqjz" outputId="96b9644f-0a55-4d3d-b93d-ee067ce5619d"
a_cut = 0.3
predictions.loc[predictions.prob_single < a_cut, "label"] = "double-cropped"
predictions.loc[predictions.prob_single >= a_cut, "label"] = "single-cropped"
# predictions["label"] = predictions["label"].astype(int)
predictions.head(2)
# -

predictions["label"].value_counts()

# +
plot_dir = path_to_shpfile + "plots/"
os.makedirs(plot_dir, exist_ok=True)

# Pick a field
an_ID = list(predictions.ID.unique())[3]

for an_ID in list(predictions.ID.unique()):
    a_field = regular_df[regular_df.ID==an_ID].copy()
    a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                           sharex='col', sharey='row',
                           gridspec_kw={'hspace': 0.2, 'wspace': .05});
    ax.grid(True);
    ax.plot(a_field['human_system_start_time'], a_field[VI_idx],
            linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
            label=f"smooth {VI_idx}")
    
    # Raw data where we started from
    raw = reduced[reduced.ID==an_ID].copy()
    raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
    ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");
    
    crop_ = predictions[predictions.ID == an_ID]["CropTyp"].values[0]
    label_ = predictions[predictions.ID==an_ID]["label"].values[0]
    county_ = predictions[predictions.ID==an_ID]["county"].values[0]
    Irrigation_ = predictions[predictions.ID==an_ID]["Irrigtn"].values[0]
    
    ax.set_title(crop_ + ", " + label_ + ", " + county_ + ", " + Irrigation_)
    ax.legend(loc="lower right");
    plt.ylim([-0.5, 1.2]);
    
    file_name = plot_dir + crop_.replace(" ", "").replace(",", "_") + "_"+ an_ID + ".pdf"
    plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);
    plt.close()

# + colab={"base_uri": "https://localhost:8080/", "height": 314} executionInfo={"elapsed": 851, "status": "ok", "timestamp": 1702075971725, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="G8Rd7Rdr99aQ" outputId="9c7a79ec-6774-4a92-f0ea-3ec5b5ee312b"
#  Pick a field
an_ID = IDs[3]
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], a_field[VI_idx],
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
        label=f"smooth {VI_idx}")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");

crop_ = predictions[predictions.ID == an_ID]["CropTyp"].values[0]
if model=="DL":
  label_ = predictions[predictions.ID==an_ID]["label"].values[0]
else:
  ss = model + "_" + "VI_idx" + "_preds"
  label_ = list(predictions.loc[predictions.ID==an_ID, ss])[0]
  label_ = f"SVM prediction is {label_}."

ax.set_title(crop_ + ", " + label_)
ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# + [markdown] id="e2_PZfWI-P-D"
# ##### Export predictions to Google Drive!

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1702075971725, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="LaNYsp1z-S8K" outputId="69613e34-d9d1-47ed-c633-d5da7d343c3b"
pred_dir = data_base + "joel_data/predicted_classes/"
out_fileName = shapefile_name + \
                start_date.replace("-", "_") + \
                end_date.replace("-", "_") + "_batch" + str(batch_number)  + ".csv"

out_name = pred_dir + out_fileName
predictions.to_csv(out_name, index=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1702075971725, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="Gal_QpyU_GTy" outputId="25b96760-ecdd-451b-f9d5-91a39b5767b3"
from datetime import datetime

out_fileName = shapefile_name + \
                start_date.replace("-", "_") + \
                end_date.replace("-", "_") + "_batch" + str(batch_number) + ".sav"

out_name = pred_dir + out_fileName

export_ = {"predictions": predictions,
           "source_code" : "joel_pipeline.ipynb",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           "Present": "Amin, HN, Joel, Kirti",
           "Meeting": "Recorded on Zoom: https://www.youtube.com/watch?v=FuBz4IIqw7Y&ab_channel=HosseinNoorazar"}

pickle.dump(export_, open(out_name, 'wb'))

# + colab={"base_uri": "https://localhost:8080/", "height": 864} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1702075971725, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="ps2kFKIsbnMC" outputId="1eb024eb-9abe-48dd-cc55-cc9e6fe91353"
predictions.head(2)

# + colab={"base_uri": "https://localhost:8080/", "height": 314} executionInfo={"elapsed": 944, "status": "ok", "timestamp": 1702076221855, "user": {"displayName": "Hossein Noorazar", "userId": "07953809792932992694"}, "user_tz": 480} id="zMPwwNY-mqO8" outputId="a3a377b9-2a55-4a5b-aeb7-39e757e16061"
#  Pick a field
an_ID = regular_df.ID[1]
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], a_field[VI_idx],
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
        label=f"smooth {VI_idx}")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");

crop_ = predictions[predictions.ID == an_ID]["CropTyp"].values[0]
if model=="DL":
  label_ = predictions[predictions.ID==an_ID]["label"].values[0]
else:
  ss = model + "_" + "VI_idx" + "_preds"
  label_ = list(predictions.loc[predictions.ID==an_ID, ss])[0]
  label_ = f"SVM prediction is {label_}."

ax.set_title(crop_ + ", " + label_)
ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# + id="XhkK5t2FpG_z"
out_name
# -


