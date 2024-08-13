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
def dict_to_df(master_dictionary):
    # Extract all unique first and second keys
    strata = set()
    columns = set()

    for key in master_dictionary.keys():
        strata.add(key[0])
        columns.add(key[1])

    # Convert to sorted lists
    strata = sorted(strata)
    columns = sorted(columns)

    # Create a list to store each row as a dictionary
    rows = []

    # Populate the list with rows
    for s in strata:
        row = {"strata": s}
        for c in columns:
            row[c] = master_dictionary.get((s, c), [None])[
                0
            ]  # Get the first value from the list or None if key doesn't exist
        rows.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows, columns=["strata"] + columns)
    return df

# # Function to calculate strata pool numbers and areas based on paper formulas
# def strata_pools(test_df, id_dict, acr_data):
#     master_dict = defaultdict(list)
#     for strata in test_df["CropTyp"].unique():
#         strata_subset = {key: value for key, value in id_dict.items() if key[1] == strata}
#         A_n_star_h_list = [
#             value[2] for key, values in strata_subset.items() for value in values
#         ]
#         A_n_star_h = sum(A_n_star_h_list)

#         index = acr_data[acr_data["CropTyp"] == strata].index[0]
#         # Now use .at to access the specific value
#         A_N_star_h = acr_data.at[index, "denom_acr"]
#         N_star_h = acr_data.at[index, "denom"]
#         n_star_h = len(A_n_star_h_list)

#         master_dict[(strata, "n_star_h")].append(n_star_h)
#         master_dict[(strata, "A_n_star_h")].append(A_n_star_h)
#         master_dict[(strata, "A_N_star_h")].append(A_N_star_h)
#         master_dict[(strata, "N_star_h")].append(N_star_h)
#     return master_dict


# -

def calculate_metrics(test_df, acr_data, N, A_N):

    id_dict = defaultdict(list)
    # Since prediction columns name is different for different algorithms
    pred_col_name = next((col for col in test_df.columns if "NDVI" in col), None)

    for idx, row in test_df.iterrows():
        id_dict[(row["Vote"], row[pred_col_name]), row["CropTyp"]].append(
            (row["ID"], row["inclusion_prob"], row["ExctAcr"])
        )

    # Correctly classified acres
    s_s_area = test_df.loc[
        (test_df["Vote"] == 1) & (test_df[pred_col_name] == 1)]["ExctAcr"].sum()
    d_d_area = test_df.loc[
        (test_df["Vote"] == 2) & (test_df[pred_col_name] == 2)]["ExctAcr"].sum()

    print("Area of correctly classified as single: ", s_s_area)
    print("Area of correctly classified as double: ", d_d_area)

    ###########################################################
    ###########################################################
    # OVERALL ACCURACY
    ###########################################################
    ###########################################################
    # You see A_ because we are using area not just counts

    master_dict = defaultdict(list)
    # Numbers of strata
    for strata in test_df["CropTyp"].unique():
        strata_subset = {key: value for key, value in id_dict.items() if key[1] == strata}
        A_n_star_h_list = [
            value[2] for key, values in strata_subset.items() for value in values
        ]
        A_n_star_h = sum(A_n_star_h_list)

        index = acr_data[acr_data["CropTyp"] == strata].index[0]
        # Now use .at to access the specific value
        A_N_star_h = acr_data.at[index, "denom_acr"]
        N_star_h = acr_data.at[index, "denom"]
        n_star_h = len(A_n_star_h_list)

        master_dict[(strata, "n_star_h")].append(n_star_h)
        master_dict[(strata, "A_n_star_h")].append(A_n_star_h)
        master_dict[(strata, "A_N_star_h")].append(A_N_star_h)
        master_dict[(strata, "N_star_h")].append(N_star_h)

        A_yu_list = [
            value[2]
            for key, values in strata_subset.items()
            for value in values
            if key[0][0] == key[0][1]
        ]
        A_yu = sum(A_yu_list)

        y_bar_h = A_yu / A_n_star_h

        # Sample variance (based on counts not area)
        y_bar_h_count = len(A_yu_list) / master_dict[(strata, "n_star_h")][0]
        yu_0_1 = np.append(np.ones(len(A_yu_list)), np.zeros(n_star_h - len(A_yu_list)))
        sy_h_2 = sum((yu_0_1 - y_bar_h_count) ** 2 / master_dict[(strata, "n_star_h")][0])

        master_dict[strata, "y_bar_h"].append(y_bar_h)
        master_dict[strata, "sy_h_2"].append(sy_h_2)

    master_df = dict_to_df(master_dict)
    master_df = master_df.dropna()

    Y_bar_list = []
    v_list = []
    v_list_countbased = []
    for strata in master_df["strata"].unique():
        A_N_star_h = master_df.loc[master_df["strata"] == strata, "A_N_star_h"].values[0]
        A_n_star_h = master_df.loc[master_df["strata"] == strata, "A_n_star_h"].values[0]
        sy_h_2 = master_df.loc[master_df["strata"] == strata, "sy_h_2"].values[0]
        y_bar_h = master_df.loc[master_df["strata"] == strata, "y_bar_h"].values[0]

        Y_bar_list.append(A_N_star_h * y_bar_h)

        v_list.append(A_N_star_h**2 * (1 - A_n_star_h / A_N_star_h) * sy_h_2 / A_n_star_h)

        # v_list_countbased.append(
        #     N_star_h**2
        #     * (1 - n_star_h_dict[strata][0] / N_star_h)
        #     * s_yh_2_dict[strata][0]
        #     / n_star_h_dict[strata][0]
        # )

    Overall_acc = sum(Y_bar_list) / A_N
    print(sum(Y_bar_list))
    print(A_N)
    print("Overall Accuracy = ", Overall_acc)

    # Variance of overall accuracy
    v_o = (1 / (N)) * sum(v_list)
    se_o = np.sqrt(v_o)
    # v_o_countbased = (1 / N**2) * sum(v_list_countbased)
    print("Area-based standard error of overall accuracy = ", se_o)
    # print("Count-based Variance of overall accuracy = ", v_o_countbased)

    ######################################################################
    ######################################################################
    # USER'S ACCURACY AND SE
    ######################################################################
    ######################################################################
    for c in [1, 2]:
        # Filter for instances that are mapped as c.
        c_dict = {key: value for key, value in id_dict.items() if key[0][0] == c}
        # Filter for instances that are mapped as c and referenced as c, too (cc).
        cc_dict = {
            key: value
            for key, value in id_dict.items()
            if (key[0][0] == c and key[0][1] == c)
        }

        # List stratas for c and cc
        # X
        c_strata_list = [key[1] for key, _ in c_dict.items()]
        # Y
        cc_strata_list = [key[1] for key, _ in cc_dict.items()]

        # ##### Calculate numerator sum

        master_dict = defaultdict(list)
        # Numbers of strata
        for strata in test_df["CropTyp"].unique():
            strata_subset = {
                key: value for key, value in id_dict.items() if key[1] == strata
            }
            A_n_star_h_list = [
                value[2] for key, values in strata_subset.items() for value in values
            ]
            A_n_star_h = sum(A_n_star_h_list)

            index = acr_data[acr_data["CropTyp"] == strata].index[0]
            # Now use .at to access the specific value
            A_N_star_h = acr_data.at[index, "denom_acr"]
            N_star_h = acr_data.at[index, "denom"]

            master_dict[(strata, "n_star_h")].append(len(A_n_star_h_list))
            master_dict[(strata, "A_n_star_h")].append(A_n_star_h)
            master_dict[(strata, "A_N_star_h")].append(A_N_star_h)
            master_dict[(strata, "N_star_h")].append(N_star_h)

        for strata in np.unique(np.array(cc_strata_list)):
            strata_subset = {
                key: value for key, value in cc_dict.items() if key[1] == strata
            }

            A_yu_list = [
                value[2]
                for key, values in strata_subset.items()
                for value in values
                if key[0][0] == key[0][1]
            ]
            yu_IDs = np.array(
                [
                    value[0]
                    for key, values in strata_subset.items()
                    for value in values
                    if key[0][0] == key[0][1]
                ]
            )
            A_yu = sum(A_yu_list)

            # Sample variance (based on counts not area)
            y_bar_h_count = len(A_yu_list) / master_dict[(strata, "n_star_h")][0]
            sy_h_2 = (len(A_yu_list) - y_bar_h_count) ** 2 / master_dict[
                (strata, "n_star_h")
            ][0]

            master_dict[(strata, "n_yu")].append(len(A_yu_list))
            master_dict[(strata, "yu_IDs")].append(yu_IDs)
            master_dict[(strata, "y_bar_h")].append(
                A_yu / master_dict[(strata, "A_n_star_h")][0]
            )
            master_dict[(strata, "y_bar_h_count")].append(y_bar_h_count)
            master_dict[(strata, "sy_h_2")].append(sy_h_2)
            master_dict[(strata, "Y_bar")].append(
                master_dict[(strata, "A_N_star_h")][0] * master_dict[(strata, "y_bar_h")][0]
            )
            # master_dict[(strata, "v_y_list")].append(
            #     A_N_star_h**2
            #     * (1 - A_n_star_h_dict[strata][0] / A_N_star_h)
            #     * master_dict[(strata, "sy_h_2")][0]
            #     / master_dict[(strata, "A_n_star_h")][0]
            # )

        ###########  Calculate denominator sum  ###########
        for strata in np.unique(np.array(c_strata_list)):
            strata_subset = {
                key: value for key, value in c_dict.items() if key[1] == strata
            }

            A_xu_list = [
                value[2] for key, values in strata_subset.items() for value in values
            ]
            xu_IDs = np.array(
                [value[0] for key, values in strata_subset.items() for value in values]
            )
            A_xu = sum(A_xu_list)

            # Sample variance (based on counts not area)
            x_bar_h_count = len(A_xu_list) / master_dict[(strata, "n_star_h")][0]
            sx_h_2 = (len(A_xu_list) - x_bar_h_count) ** 2 / master_dict[
                (strata, "n_star_h")
            ][0]

            master_dict[(strata, "n_xu")].append(len(A_xu_list))
            master_dict[(strata, "xu_IDs")].append(xu_IDs)
            master_dict[(strata, "x_bar_h")].append(
                A_xu / master_dict[(strata, "A_n_star_h")][0]
            )
            master_dict[(strata, "x_bar_h_count")].append(x_bar_h_count)
            master_dict[(strata, "sx_h_2")].append(sx_h_2)
            master_dict[(strata, "X_bar")].append(
                master_dict[(strata, "A_N_star_h")][0] * master_dict[(strata, "x_bar_h")][0]
            )
            # master_dict[(strata, "v_x_list")].append(
            #     A_N_star_h_x**2
            #     * (1 - A_n_star_h_dict[strata][0] / A_N_star_h)
            #     * master_dict[(strata, "sy_h_2")][0]
            #     / master_dict[(strata, "A_n_star_h")][0]
            # )

        master_dict = {key: master_dict[key] for key in sorted(master_dict.keys())}
        master_dict = defaultdict(list, master_dict)

        # put yu and xu of 0 - 1s in the master dict
        xu_id = {
            key[0]: np.array(sorted(value))
            for key, values in master_dict.items()
            for value in values
            if key[1] == "xu_IDs"
        }
        yu_id = {
            key[0]: np.array(sorted(value))
            for key, values in master_dict.items()
            for value in values
            if key[1] == "yu_IDs"
        }

        for key, value in xu_id.items():
            if key not in yu_id:
                master_dict[(key, "yu_0_1")].append(np.zeros(len(xu_id[key])))
            else:
                yu_in_xu_0_1 = np.array((np.isin(xu_id[key], yu_id[key])).astype(int))
                master_dict[(key, "xu_0_1")].append(np.ones(len(yu_in_xu_0_1)))
                master_dict[(key, "yu_0_1")].append(yu_in_xu_0_1)

        master_dict = {key: master_dict[key] for key in sorted(master_dict.keys())}
        master_dict = defaultdict(list, master_dict)

        # Convert master_dict to a dataframe
        master_df = dict_to_df(master_dict)
        master_df = master_df.dropna()

        # Calculate s_xy_h
        for strata in master_df["strata"].unique():
            yu = master_df.loc[master_df["strata"] == strata, "yu_0_1"].values[0]
            xu = master_df.loc[master_df["strata"] == strata, "xu_0_1"].values[0]
            ybar_h = master_df.loc[master_df["strata"] == strata, "y_bar_h_count"].values[0]
            xbar_h = master_df.loc[master_df["strata"] == strata, "x_bar_h_count"].values[0]
            n_star_h = master_df.loc[master_df["strata"] == strata, "n_star_h"].values[0]

            s_xy_h = sum((yu - ybar_h) * (xu - xbar_h) / n_star_h - 1)
            master_df.loc[master_df["strata"] == strata, "s_xy_h"] = s_xy_h

            # Calculate X_hat
            A_N_star_h = master_df.loc[master_df["strata"] == strata, "A_N_star_h"].values[
                0
            ]
            x_hat = A_N_star_h * xbar_h
            master_df.loc[master_df["strata"] == strata, "x_hat"] = x_hat

        # Calculate user's accuracy
        Y_bar_list = [value[0] for key, value in master_dict.items() if key[1] == "Y_bar"]
        numerator_sum = sum(Y_bar_list)

        X_bar_list = [value[0] for key, value in master_dict.items() if key[1] == "X_bar"]
        denominator_sum = sum(X_bar_list)

        users_acc = numerator_sum / denominator_sum
        print("Class: ", c)
        print((numerator_sum, denominator_sum))
        print("Area-based user's accuracy = ", users_acc)

        # Calculate variance of user's accuracy
        v_sum_list = []
        for strata in master_df["strata"].unique():
            A_N_star_h = master_df.loc[master_df["strata"] == strata, "A_N_star_h"].values[
                0
            ]
            A_n_star_h = master_df.loc[master_df["strata"] == strata, "A_n_star_h"].values[
                0
            ]
            sy_h_2 = master_df.loc[master_df["strata"] == strata, "sy_h_2"].values[0]
            sx_h_2 = master_df.loc[master_df["strata"] == strata, "sx_h_2"].values[0]
            s_xy_h = master_df.loc[master_df["strata"] == strata, "s_xy_h"].values[0]

            v_sum_list.append(
                A_N_star_h**2
                * (1 - A_n_star_h / A_N_star_h)
                * (sy_h_2 + users_acc**2 * sx_h_2 - 2 * users_acc * s_xy_h)
                / A_n_star_h
            )

        v_u = (1 / master_df["x_hat"].sum()) * sum(v_sum_list)
        se_u = np.sqrt(v_u)
        print("Area-based standard error of user's accuracy = ", se_u)

        ######################################################################
        ######################################################################
        # PRODUCER'S ACCURACY AND SE
        ######################################################################
        ######################################################################

        # Filter for instances that are mapped as c.
        c_dict = {key: value for key, value in id_dict.items() if key[0][1] == c}
        # Filter for instances that are mapped as c and referenced as c, too (cc).
        cc_dict = {
            key: value
            for key, value in id_dict.items()
            if (key[0][0] == c and key[0][1] == c)
        }

        # List stratas for c and cc
        # X
        c_strata_list = [key[1] for key, _ in c_dict.items()]
        # Y
        cc_strata_list = [key[1] for key, _ in cc_dict.items()]

        # ##### Calculate numerator sum

        master_dict = defaultdict(list)
        # Numbers of strata
        for strata in test_df["CropTyp"].unique():
            strata_subset = {
                key: value for key, value in id_dict.items() if key[1] == strata
            }
            A_n_star_h_list = [
                value[2] for key, values in strata_subset.items() for value in values
            ]
            A_n_star_h = sum(A_n_star_h_list)

            index = acr_data[acr_data["CropTyp"] == strata].index[0]
            # Now use .at to access the specific value
            A_N_star_h = acr_data.at[index, "denom_acr"]
            N_star_h = acr_data.at[index, "denom"]

            master_dict[(strata, "n_star_h")].append(len(A_n_star_h_list))
            master_dict[(strata, "A_n_star_h")].append(A_n_star_h)
            master_dict[(strata, "A_N_star_h")].append(A_N_star_h)
            master_dict[(strata, "N_star_h")].append(N_star_h)

        for strata in np.unique(np.array(cc_strata_list)):
            strata_subset = {
                key: value for key, value in cc_dict.items() if key[1] == strata
            }

            A_yu_list = [
                value[2]
                for key, values in strata_subset.items()
                for value in values
                if key[0][0] == key[0][1]
            ]
            yu_IDs = np.array(
                [
                    value[0]
                    for key, values in strata_subset.items()
                    for value in values
                    if key[0][0] == key[0][1]
                ]
            )
            A_yu = sum(A_yu_list)

            # Sample variance (based on counts not area)
            y_bar_h_count = len(A_yu_list) / master_dict[(strata, "n_star_h")][0]
            sy_h_2 = (len(A_yu_list) - y_bar_h_count) ** 2 / master_dict[
                (strata, "n_star_h")
            ][0]

            master_dict[(strata, "n_yu")].append(len(A_yu_list))
            master_dict[(strata, "yu_IDs")].append(yu_IDs)
            master_dict[(strata, "y_bar_h")].append(
                A_yu / master_dict[(strata, "A_n_star_h")][0]
            )
            master_dict[(strata, "y_bar_h_count")].append(y_bar_h_count)
            master_dict[(strata, "sy_h_2")].append(sy_h_2)
            master_dict[(strata, "Y_bar")].append(
                master_dict[(strata, "A_N_star_h")][0] * master_dict[(strata, "y_bar_h")][0]
            )
            # master_dict[(strata, "v_y_list")].append(
            #     A_N_star_h**2
            #     * (1 - A_n_star_h_dict[strata][0] / A_N_star_h)
            #     * master_dict[(strata, "sy_h_2")][0]
            #     / master_dict[(strata, "A_n_star_h")][0]
            # )

        ###########  Calculate denominator sum  ###########
        for strata in np.unique(np.array(c_strata_list)):
            strata_subset = {
                key: value for key, value in c_dict.items() if key[1] == strata
            }

            A_xu_list = [
                value[2] for key, values in strata_subset.items() for value in values
            ]
            xu_IDs = np.array(
                [value[0] for key, values in strata_subset.items() for value in values]
            )
            A_xu = sum(A_xu_list)

            # Sample variance (based on counts not area)
            x_bar_h_count = len(A_xu_list) / master_dict[(strata, "n_star_h")][0]
            sx_h_2 = (len(A_xu_list) - x_bar_h_count) ** 2 / master_dict[
                (strata, "n_star_h")
            ][0]

            master_dict[(strata, "n_xu")].append(len(A_xu_list))
            master_dict[(strata, "xu_IDs")].append(xu_IDs)
            master_dict[(strata, "x_bar_h")].append(
                A_xu / master_dict[(strata, "A_n_star_h")][0]
            )
            master_dict[(strata, "x_bar_h_count")].append(x_bar_h_count)
            master_dict[(strata, "sx_h_2")].append(sx_h_2)
            master_dict[(strata, "X_bar")].append(
                master_dict[(strata, "A_N_star_h")][0] * master_dict[(strata, "x_bar_h")][0]
            )
            # master_dict[(strata, "v_x_list")].append(
            #     A_N_star_h_x**2
            #     * (1 - A_n_star_h_dict[strata][0] / A_N_star_h)
            #     * master_dict[(strata, "sy_h_2")][0]
            #     / master_dict[(strata, "A_n_star_h")][0]
            # )

        master_dict = {key: master_dict[key] for key in sorted(master_dict.keys())}
        master_dict = defaultdict(list, master_dict)

        # put yu and xu of 0 - 1s in the master dict
        xu_id = {
            key[0]: np.array(sorted(value))
            for key, values in master_dict.items()
            for value in values
            if key[1] == "xu_IDs"
        }
        yu_id = {
            key[0]: np.array(sorted(value))
            for key, values in master_dict.items()
            for value in values
            if key[1] == "yu_IDs"
        }

        for key, value in xu_id.items():
            if key not in yu_id:
                master_dict[(key, "yu_0_1")].append(np.zeros(len(xu_id[key])))
            else:
                yu_in_xu_0_1 = np.array((np.isin(xu_id[key], yu_id[key])).astype(int))
                master_dict[(key, "xu_0_1")].append(np.ones(len(yu_in_xu_0_1)))
                master_dict[(key, "yu_0_1")].append(yu_in_xu_0_1)

        master_dict = {key: master_dict[key] for key in sorted(master_dict.keys())}
        master_dict = defaultdict(list, master_dict)

        # Convert master_dict to a dataframe
        master_df = dict_to_df(master_dict)
        master_df = master_df.dropna()

        # Calculate s_xy_h
        for strata in master_df["strata"].unique():
            yu = master_df.loc[master_df["strata"] == strata, "yu_0_1"].values[0]
            xu = master_df.loc[master_df["strata"] == strata, "xu_0_1"].values[0]
            ybar_h = master_df.loc[master_df["strata"] == strata, "y_bar_h_count"].values[0]
            xbar_h = master_df.loc[master_df["strata"] == strata, "x_bar_h_count"].values[0]
            n_star_h = master_df.loc[master_df["strata"] == strata, "n_star_h"].values[0]

            s_xy_h = sum((yu - ybar_h) * (xu - xbar_h) / n_star_h - 1)
            master_df.loc[master_df["strata"] == strata, "s_xy_h"] = s_xy_h

            # Calculate X_hat
            A_N_star_h = master_df.loc[master_df["strata"] == strata, "A_N_star_h"].values[
                0
            ]
            x_hat = A_N_star_h * xbar_h
            master_df.loc[master_df["strata"] == strata, "x_hat"] = x_hat

        # Calculate user's accuracy
        Y_bar_list = [value[0] for key, value in master_dict.items() if key[1] == "Y_bar"]
        numerator_sum = sum(Y_bar_list)

        X_bar_list = [value[0] for key, value in master_dict.items() if key[1] == "X_bar"]
        denominator_sum = sum(X_bar_list)

        producers_acc = numerator_sum / denominator_sum
        print((numerator_sum, denominator_sum))
        print("Area-based user's producer's = ", producers_acc)

        # Calculate variance of user's accuracy
        v_sum_list = []
        for strata in master_df["strata"].unique():
            A_N_star_h = master_df.loc[master_df["strata"] == strata, "A_N_star_h"].values[
                0
            ]
            A_n_star_h = master_df.loc[master_df["strata"] == strata, "A_n_star_h"].values[
                0
            ]
            sy_h_2 = master_df.loc[master_df["strata"] == strata, "sy_h_2"].values[0]
            sx_h_2 = master_df.loc[master_df["strata"] == strata, "sx_h_2"].values[0]
            s_xy_h = master_df.loc[master_df["strata"] == strata, "s_xy_h"].values[0]

            v_sum_list.append(
                A_N_star_h**2
                * (1 - A_n_star_h / A_N_star_h)
                * (sy_h_2 + users_acc**2 * sx_h_2 - 2 * users_acc * s_xy_h)
                / A_n_star_h
            )

        v_p = (1 / master_df["x_hat"].sum()) * sum(v_sum_list)
        se_p = np.sqrt(v_p)
        print("Area-based standard error of producer's accuracy = ", se_p)

        return {'o': Overall_acc,
                'se_o': se_o,
                'user': users_acc,
                'se_u': se_u, 
                'p': producers_acc,
                'se_p': se_p,
                'id_dict':id_dict}


list(data["six_OverSam_TestRes"].keys())[:4]

data["six_OverSam_TestRes"]["test_results_SVM"].keys()

data["six_OverSam_TestRes"]["test_results_SVM"]["train_ID0"].keys()

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
acr_data = data["six_OverSam_TestRes"]["inclusion_prob"]
field_info = data["field_info"][["ID", "ExctAcr"]]


A_N = data["six_OverSam_TestRes"]["inclusion_prob"]["denom_acr"].sum()
N = sum(data["six_OverSam_TestRes"]["inclusion_prob"]["denom"])

metrics_list = []
for key, value in dict(list(data["six_OverSam_TestRes"].items())[:4]).items():
    print(key)
    test_sets = data["six_OverSam_TestRes"][key]
    for key_, value_ in test_sets.items():
        print(key_)
        test_set = test_sets[key_]["a_test_set_df"]
        prob = data["six_OverSam_TestRes"]["inclusion_prob"]
        test_set = test_set.merge(prob, on="CropTyp", how="right")
        test_set = test_set.merge(field_info, on="ID", how="inner")

        metrics = calculate_metrics(test_set, acr_data, N, A_N)
        metrics_list.append({
        'model': key,
        'test_set': key_,
        'Overall_acc': metrics['o'],
        'SE_O': metrics['se_o'],
        'User\'s_acc': metrics['user'],
        'SE_U': metrics['se_u'],
        'Producer\'s_acc': metrics['p'],
        'SE_P': metrics['se_p']
    })

df = pd.DataFrame(metrics_list)
df
# -

A_N

metrics['id_dict']
