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

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import defaultdict

# +
# Ubuntu
# path_to_data = (
#     "/home/amnnrz/OneDrive - a.norouzikandelati/"
#     "Ph.D/Projects/Double_Crop_Mapping/"
# )

# Mac
path_to_data = ('/Users/aminnorouzi/Library/CloudStorage/'
                'OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/'
                'Projects/Double_Crop_Mapping/')

file_path = path_to_data + "five_OverSam_TestRes_and_InclusionProb.sav"
test_data = pd.read_pickle(file_path)
field_info = test_data["field_info"][["ID", "ExctAcr"]]
test_set = test_data['five_OverSam_TestRes']['test_results_DL']['train_ID1']['a_test_set_df']
test_set.rename(columns={"Vote": "reference", "NDVI_SG_DL_p3": "map"}, inplace=True)
cm = pd.crosstab(test_set["map"], test_set["reference"])


prob = test_data["five_OverSam_TestRes"]["inclusion_prob"]
test_set = test_set.merge(prob, on="CropTyp", how="right")
test_set = test_set.merge(field_info, on="ID", how="inner")
test_set

id_dict = defaultdict(list)
for idx, row in test_set.iterrows():
    id_dict[(row["reference"], row["map"])].append((row['ID'],
     row['inclusion_prob'], row['ExctAcr']))

# for cell, items in id_dict.items():
# -

cm

id_dict
