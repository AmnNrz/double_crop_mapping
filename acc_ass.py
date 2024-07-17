# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: tillmap
#     language: python
#     name: python3
# ---

# # Accuracy assessment of double cropping paper
# This notebook is based on the methodologies described in the following paper:
#
# Stehman, Stephen V. "Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes." International Journal of Remote Sensing 35.13 (2014): 4923-4939.

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import defaultdict

# +
path_to_data = (
    "/home/amnnrz/OneDrive - a.norouzikandelati/Ph.D/Projects/Double_Crop_Mapping/"
)

# path_to_data = (
#     "/Users/aminnorouzi/Library/CloudStorage/"
#     "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
#     "Projects/Double_Crop_Mapping/"
# )

file_path = path_to_data + "five_OverSam_TestRes_and_InclusionProb.sav"
test_data = pd.read_pickle(file_path)
field_info = test_data["field_info"][["ID", "ExctAcr"]]
test_set = test_data['five_OverSam_TestRes']['test_results_DL']['train_ID1']['a_test_set_df']
cm = confusion_matrix(test_set['NDVI_SG_DL_p3'], test_set['Vote'])


prob = test_data["five_OverSam_TestRes"]["inclusion_prob"]
test_set = test_set.merge(prob, on="CropTyp", how="right")
test_set = test_set.merge(field_info, on="ID", how="inner")
test_set

id_dict = defaultdict(list)
for idx, row in test_set.iterrows():
    id_dict[(row["Vote"], row["NDVI_SG_DL_p3"]), row["CropTyp"]].append((row['ID'],
     row['inclusion_prob'], row['ExctAcr']))

# -

test_data['five_OverSam_TestRes']['inclusion_prob']

test_set

# ### Formula to calculate overall accuracy
# ![Overal_acc](formulas/Unbiased_estimator_0.png)
# ![Overal_acc](formulas/Unbiased_estimator_1.png)

# ### Overall accuracy

strata_err_mat

total_A = 0
for key, value in strata_subset.items():
    print([val[2] for val in value])

# +
strata_err_mat = defaultdict(list)
for strata in test_set['CropTyp'].unique():
    strata_subset = {key: value for key, value in id_dict.items() if key[1] == strata}

   
    total_A = 0
    for key, value in strata_subset.items():
        total_A += np.array([val[2] for val in value]).sum()
    total_A

   
    strata_size = 0
    for key, value in strata_subset.items():
        strata_size += len([val[2] for val in value])
        Aij = np.array([val[2] for val in value]).sum()
        pij = Aij/total_A # pij : propotion of area (this is not the same as
                        # pij in Foody paper which is error matrix values)
        strata_err_mat[key[0]].append((key[1], pij, strata_size))

N = test_set.shape[0]
for key, value in strata_err_mat.items():
    N_star_h = np.array([i[2] for i in value])
    y_bar_h = np.array([i[1] for i in value])
    strata_err_mat[key].append(sum(N_star_h * y_bar_h/N))

    # To be used as numerators for user's and producer's accuracy
    strata_err_mat[key].append(sum(N_star_h * y_bar_h))

# Extracting the last element of each list for each key
error_mat_dict = {(k1, k2): v[-1] for (k1, k2), v in strata_err_mat.items()}

# Creating a DataFrame from the extracted data
df = pd.DataFrame(list(error_mat_dict.items()), columns=["keys", "values"])

# Splitting the keys into separate columns
df[["Map", "Reference"]] = pd.DataFrame(df["keys"].tolist(), index=df.index)

# Creating cross tab
error_matrix = pd.crosstab(
    index=df["Map"], columns=df["Reference"], values=df["values"], aggfunc="sum"
)

# overall_accuray 
diag_vals = [] 
for i in np.arange(0, error_matrix.shape[0]):
    diag_vals.append(error_matrix.iloc[i, i])
overall_acc = np.array(diag_vals).sum()/error_matrix.values.sum()
print(overall_acc)
# -

# ### User's accuracy

# ![Overal_acc](formulas/Users_producers.png)

# +
c = 1    # class = [1, 2]
numor = strata_err_mat[(c, c)][-1]    # Get the numerator

# Filter for map class c (rows == c)
filtered_dict = {key: values[:-2] for key, values in strata_err_mat.items() if 
                 key[0] == c}
# Transform filtered_dict : {"strata": (area proportion, count)}
transformed_dict = {}

# Iterate through the original dictionary
for key, value_list in filtered_dict.items():
    for item in value_list:
        # print(item)
        strata, number, count = item
        
        if strata not in transformed_dict:
            transformed_dict[strata] = (number, count)
        else:
            transformed_dict[strata] = (transformed_dict[strata][0] + number,
                                         transformed_dict[strata][1] + count)

N_star_h = np.array([value[1] for key, value in transformed_dict.items()])
y_bar_h = np.array([value[0] for key, value in transformed_dict.items()])

denomor = sum(N_star_h * y_bar_h)

users_acc = numor/denomor
users_acc

# -

# ### Producer's accuracy

# ![Overal_acc](formulas/Users_producers.png)

# +
# Filter for reference class c (columns == c)
filtered_dict = {key: values[:-2] for key, values in strata_err_mat.items() if 
                 key[1] == c}
# Transform filtered_dict : {"strata": (area proportion, count)}
transformed_dict = {}

# Iterate through the original dictionary
for key, value_list in filtered_dict.items():
    for item in value_list:
        # print(item)
        strata, number, count = item
        
        if strata not in transformed_dict:
            transformed_dict[strata] = (number, count)
        else:
            transformed_dict[strata] = (transformed_dict[strata][0] + number,
                                         transformed_dict[strata][1] + count)

N_star_h = np.array([value[1] for key, value in transformed_dict.items()])
y_bar_h = np.array([value[0] for key, value in transformed_dict.items()])

denomor = sum(N_star_h * y_bar_h)

Prods_acc = numor/denomor
Prods_acc

