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

# # Accuracy assessment of double cropping paper
# This notebook is based on the methodologies described in the following paper:
#
# Stehman, Stephen V. "Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes." International Journal of Remote Sensing 35.13 (2014): 4923-4939.

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import defaultdict



# +
# path_to_data = (
#     "/home/amnnrz/OneDrive - a.norouzikandelati/Ph.D/Projects/Double_Crop_Mapping/"
# )

path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
    "Projects/Double_Crop_Mapping/"
)

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

acr_data

# +
# You see A_ because we are using area not just counts
A_N = data['six_OverSam_TestRes']['inclusion_prob']['denom_acr'].sum()
N = sum(data["six_OverSam_TestRes"]["inclusion_prob"]["denom"])

# Calculate y_bar_h
y_bar_h_dict = defaultdict(list)
s_yh_2_dict = defaultdict(list)
A_n_star_h_dict = defaultdict(list)
n_star_h_dict = defaultdict(list)
for strata in test_set['CropTyp'].unique():
    strata_subset = {key: value for key, value in id_dict.items() if key[1] == strata}

    A_yu_list = [value[2] for key, values in strata_subset.items()
                      for value in values if key[0][0] == key[0][1]]
    A_yu = sum(A_yu_list)

    A_n_star_h_list = [value[2] for key, values in strata_subset.items()
                    for value in values]
    A_n_star_h = sum(A_n_star_h_list)
    n_star_h_dict[strata].append(len(A_n_star_h_list))
    A_n_star_h_dict[strata].append(A_n_star_h)

    y_bar_h = A_yu/A_n_star_h
    y_bar_h_dict[strata].append(y_bar_h)

    # Sample variance (based on counts not area)
    n_y_bar_h = len(A_yu_list)/len(A_n_star_h_list)
    s_yh_2_h = (len(A_yu_list) - n_y_bar_h) ** 2 / len(A_n_star_h_list)
    s_yh_2_dict[strata].append(s_yh_2_h)

acr_data = data['six_OverSam_TestRes']['inclusion_prob']
Y_bar_list = []
v_list = []
v_list_countbased = []
for strata, y_bar_h in y_bar_h_dict.items():
    # Find the index of the first row where strata is "CropTyp".
    index = acr_data[acr_data['CropTyp'] == strata].index[0]

    # Now use .at to access the specific value
    A_N_star_h = acr_data.at[index, 'denom_acr']
    N_star_h = acr_data.at[index, 'denom']

    Y_bar_list.append(A_N_star_h * y_bar_h[0])

    v_list.append(A_N_star_h**2 * (1 - A_n_star_h_dict[strata][0] / A_N_star_h) * s_yh_2_dict[
        strata
    ][0] / A_n_star_h_dict[strata][0])

    v_list_countbased.append(
        N_star_h**2
        * (1 - n_star_h_dict[strata][0] / N_star_h)
        * s_yh_2_dict[strata][0]
        / n_star_h_dict[strata][0]
    )


Overall_acc = sum(Y_bar_list)/A_N
print("Overall Accuracy = ", Overall_acc)

# Variance of overall accuracy
v_o = (1 / A_N ** 2) * sum(v_list)

v_o_countbased = (1 / N**2) * sum(v_list_countbased)
print("Area-based Variance of overall accuracy = ", v_o)
print("Count-based Variance of overall accuracy = ", v_o_countbased)
# -

s_yh_2_dict[strata]

# ### User's accuracy

id_dict

# +
c = 1 # We have two classes: 1 and 2

# Filter for instances that are mapped as c.
c_dict = {key: value for key, value in id_dict.items() if key[0][0] == c}
# Filter for instances that are mapped as c and referenced as c, too (cc).
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

# ### Producer's accuracy

# +
c = 2  # We have two classes: 1 and 2

# Filter for instances that are referenced as c.
c_dict = {key: value for key, value in id_dict.items() if key[0][1] == c}
# Filter for instances that are referenced as c and mapped as c, too.
cc_dict = {
    key: value for key, value in id_dict.items() if (key[0][0] == c and key[0][1] == c)
}

# List stratas for c and cc
c_strata_list = [key[1] for key, _ in c_dict.items()]
cc_strata_list = [key[1] for key, _ in cc_dict.items()]


# ##### Calculate numerator sum
y_bar_h_dict = defaultdict(list)
for strata in np.unique(np.array(cc_strata_list)):
    strata_subset = {key: value for key, value in cc_dict.items() if key[1] == strata}
    A_yu = sum(
        [
            value[2]
            for key, values in strata_subset.items()
            for value in values
            if key[0][0] == key[0][1]
        ]
    )
    A_n_star_h = sum(
        [value[2] for key, values in strata_subset.items() for value in values]
    )

    y_bar_h_dict[strata].append(A_yu / A_n_star_h)


acr_data = data["six_OverSam_TestRes"]["inclusion_prob"]
Y_bar_list = []
for strata, y_bar_h in y_bar_h_dict.items():
    # Find the index of the first row where "CropTyp" is "alfalfa"
    index = acr_data[acr_data["CropTyp"] == strata].index[0]

    # Now use .at to access the specific value
    A_N_star_h = acr_data.at[index, "denom_acr"]

    Y_bar_list.append(A_N_star_h * y_bar_h[0])

numerator_sum = sum(Y_bar_list)

###########  Calculate denominator sum  ###########
y_bar_h_dict = defaultdict(list)
for strata in np.unique(np.array(c_strata_list)):
    strata_subset = {key: value for key, value in c_dict.items() if key[1] == strata}
    A_yu = sum([value[2] for key, values in strata_subset.items() for value in values])
    A_n_star_h = sum(
        [value[2] for key, values in strata_subset.items() for value in values]
    )

    y_bar_h_dict[strata].append(A_yu / A_n_star_h)


acr_data = data["six_OverSam_TestRes"]["inclusion_prob"]
Y_bar_list = []
for strata, y_bar_h in y_bar_h_dict.items():
    # Find the index of the first row where "CropTyp" is "alfalfa"
    index = acr_data[acr_data["CropTyp"] == strata].index[0]

    # Now use .at to access the specific value
    A_N_star_h = acr_data.at[index, "denom_acr"]

    Y_bar_list.append(A_N_star_h * y_bar_h[0])

denominator_sum = sum(Y_bar_list)

users_acc = numerator_sum / denominator_sum
print(users_acc)
# -

# ### Variance of overall accuracy


