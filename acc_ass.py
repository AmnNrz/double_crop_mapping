# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: tillmap
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

path_to_data = (
    "/home/amnnrz/OneDrive - a.norouzikandelati/"
    "Ph.D/Projects/Double_Crop_Mapping/"
)
file_path = path_to_data + "five_OverSam_TestRes_and_InclusionProb.sav"
test_data = pd.read_pickle(file_path)
test_set = test_data['five_OverSam_TestRes']['test_results_DL']['train_ID1']['a_test_set_df']
cm = confusion_matrix(test_set['NDVI_SG_DL_p3'], test_set['Vote'])
cm 
