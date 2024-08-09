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

file_path = path_to_data + "six_OverSam_TestRes_and_InclusionProb.sav"
data = pd.read_pickle(file_path)
field_info = data["field_info"][["ID", "ExctAcr"]]
test_set = data['six_OverSam_TestRes']['test_results_DL']['train_ID1']['a_test_set_df']
cm = confusion_matrix(test_set['NDVI_SG_DL_p3'], test_set['Vote'])


prob = data["six_OverSam_TestRes"]["inclusion_prob"]
test_set = test_set.merge(prob, on="CropTyp", how="right")
test_set = test_set.merge(field_info, on="ID", how="inner")
test_set

id_dict = defaultdict(list)
for idx, row in test_set.iterrows():
    id_dict[(row["Vote"], row["NDVI_SG_DL_p3"]), row["CropTyp"]].append((row['ID'],
     row['inclusion_prob'], row['ExctAcr']))
# -

# ### Formula to calculate overall accuracy
# ![Overal_acc](formulas/Unbiased_estimator_0.png)
# ![Overal_acc](formulas/Unbiased_estimator_1.png)

# ### Overall accuracy

data['six_OverSam_TestRes']['inclusion_prob']

# +
# You see A_ because we are using area not just counts 
A_N = data['six_OverSam_TestRes']['inclusion_prob']['denom_acr'].sum()

# Calculate y_bar_h 
y_bar_h_dict = defaultdict(list)
for strata in test_set['CropTyp'].unique():
    strata_subset = {key: value for key, value in id_dict.items() if key[1] == strata}
    A_yu = sum([value[2] for key, values in strata_subset.items()
                      for value in values if key[0][0] == key[0][1]])

    A_n_star_h = sum([value[2] for key, values in strata_subset.items()
                    for value in values])

    y_bar_h_dict[strata].append(A_yu/A_n_star_h)

acr_data = data['six_OverSam_TestRes']['inclusion_prob']
Y_bar_list = []
for strata, y_bar_h in y_bar_h_dict.items():
    # Find the index of the first row where "CropTyp" is "alfalfa"
    index = acr_data[acr_data['CropTyp'] == strata].index[0]

    # Now use .at to access the specific value
    A_N_star_h = acr_data.at[index, 'denom_acr']

    Y_bar_list.append(A_N_star_h * y_bar_h[0])

Overall_acc = sum(Y_bar_list)/A_N
print(Overall_acc)
# -

# ### User's accuracy

# ![Overal_acc](formulas/Users_producers.png)

# +
c = 2 # We have two classes: 1 and 2

# Filter for instances that are mapped as c.
c_dict = {key: value for key, value in id_dict.items() if key[0][0] == c}
# Filter for instances that are mapped as c and referenced as c, too.
cc_dict = {key: value for key, value in id_dict.items() if (key[0][0] == c and key[0][1] == c)}

# List stratas for c and cc 
c_strata_list = [key[1] for key, _ in c_dict.items()]
cc_strata_list = [key[1] for key, _ in cc_dict.items()]



# ##### Calculate numerator sum
y_bar_h_dict = defaultdict(list)
for strata in np.unique(np.array(cc_strata_list)):
    strata_subset = {key: value for key, value in cc_dict.items() if key[1] == strata}
    A_yu = sum([value[2] for key, values in strata_subset.items()
                      for value in values if key[0][0] == key[0][1]])
    A_n_star_h = sum([value[2] for key, values in strata_subset.items()
                    for value in values])

    y_bar_h_dict[strata].append(A_yu/A_n_star_h)



acr_data = data['six_OverSam_TestRes']['inclusion_prob']
Y_bar_list = []
for strata, y_bar_h in y_bar_h_dict.items():
    # Find the index of the first row where "CropTyp" is "alfalfa"
    index = acr_data[acr_data['CropTyp'] == strata].index[0]

    # Now use .at to access the specific value
    A_N_star_h = acr_data.at[index, 'denom_acr']

    Y_bar_list.append(A_N_star_h * y_bar_h[0])

numerator_sum = sum(Y_bar_list)

###########  Calculate denominator sum  ###########
y_bar_h_dict = defaultdict(list)
for strata in np.unique(np.array(c_strata_list)):
    strata_subset = {key: value for key, value in c_dict.items() if key[1] == strata}
    A_yu = sum([value[2] for key, values in strata_subset.items()
                      for value in values])
    A_n_star_h = sum([value[2] for key, values in strata_subset.items()
                    for value in values])

    y_bar_h_dict[strata].append(A_yu/A_n_star_h)



acr_data = data['six_OverSam_TestRes']['inclusion_prob']
Y_bar_list = []
for strata, y_bar_h in y_bar_h_dict.items():
    # Find the index of the first row where "CropTyp" is "alfalfa"
    index = acr_data[acr_data['CropTyp'] == strata].index[0]

    # Now use .at to access the specific value
    A_N_star_h = acr_data.at[index, 'denom_acr']

    Y_bar_list.append(A_N_star_h * y_bar_h[0])

denominator_sum = sum(Y_bar_list)

users_acc = numerator_sum/denominator_sum
print(users_acc)

# +
c = 2 # We have two classes: 1 and 2

# Filter for instances that are mapped as c.
c_dict = {key: value for key, value in id_dict.items() if key[0][1] == c}
# Filter for instances that are mapped as c and referenced as c, too.
cc_dict = {key: value for key, value in id_dict.items() if (key[0][0] == c and key[0][1] == c)}

# List stratas for c and cc 
c_strata_list = [key[1] for key, _ in c_dict.items()]
cc_strata_list = [key[1] for key, _ in cc_dict.items()]



# ##### Calculate numerator sum
y_bar_h_dict = defaultdict(list)
for strata in np.unique(np.array(cc_strata_list)):
    strata_subset = {key: value for key, value in cc_dict.items() if key[1] == strata}
    A_yu = sum([value[2] for key, values in strata_subset.items()
                      for value in values if key[0][0] == key[0][1]])
    A_n_star_h = sum([value[2] for key, values in strata_subset.items()
                    for value in values])

    y_bar_h_dict[strata].append(A_yu/A_n_star_h)



acr_data = data['six_OverSam_TestRes']['inclusion_prob']
Y_bar_list = []
for strata, y_bar_h in y_bar_h_dict.items():
    # Find the index of the first row where "CropTyp" is "alfalfa"
    index = acr_data[acr_data['CropTyp'] == strata].index[0]

    # Now use .at to access the specific value
    A_N_star_h = acr_data.at[index, 'denom_acr']

    Y_bar_list.append(A_N_star_h * y_bar_h[0])

numerator_sum = sum(Y_bar_list)

###########  Calculate denominator sum  ###########
y_bar_h_dict = defaultdict(list)
for strata in np.unique(np.array(c_strata_list)):
    strata_subset = {key: value for key, value in c_dict.items() if key[1] == strata}
    A_yu = sum([value[2] for key, values in strata_subset.items()
                      for value in values])
    A_n_star_h = sum([value[2] for key, values in strata_subset.items()
                    for value in values])

    y_bar_h_dict[strata].append(A_yu/A_n_star_h)



acr_data = data['six_OverSam_TestRes']['inclusion_prob']
Y_bar_list = []
for strata, y_bar_h in y_bar_h_dict.items():
    # Find the index of the first row where "CropTyp" is "alfalfa"
    index = acr_data[acr_data['CropTyp'] == strata].index[0]

    # Now use .at to access the specific value
    A_N_star_h = acr_data.at[index, 'denom_acr']

    Y_bar_list.append(A_N_star_h * y_bar_h[0])

denominator_sum = sum(Y_bar_list)

users_acc = numerator_sum/denominator_sum
print(users_acc)
# -

numerator_sum

y_bar_h_dict

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

# -

# ### Variance of overall accuracy

# S_hy =  
