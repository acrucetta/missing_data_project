import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import nan_euclidean_distances
from numba import jit

# To find out the weights following steps have to be taken:

# 1) Choose missing value to fill in the data.
# 2) Select the values in a row
# 3) Choose the number of neighbors you want to work with (ideally 2-5)
# 4) Calculate Euclidean distance from all other data points corresponding to each other in the row.
# 5) Select the smallest 2 and average out.

# Sources used:
# - https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/
# - https://www.numpyninja.com/post/mice-and-knn-missing-value-imputations-through-python
# - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html
# - https://chrisalbon.com/code/machine_learning/preprocessing_structured_data/imputing_missing_class_labels_using_k-nearest_neighbors/

def normalize_data(df):
    '''
    Normalizes the data in X

    Input:
        - df: dataframe
    Output:
        - df: normalized dataframe
    '''
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df


@jit(nopython=True)
def kSmallest(arr, k):
    '''
    Gets the k smallest values in an array

    Input:
        - arr: array
        - k: number of smallest values to return
    Output:
        - k_smallest: k smallest values in the array
    '''
    # Sort the given array
    arr_i = arr.copy()
    arr_i.sort()

    # Return k'th element in the sorted array
    # Skips the first element since it contains
    # the value 0
    return arr_i[1:k+1]


def KNN_imputation(df, Omega, keepcols, k=5):
    '''
    Recieves a dataframe df and imputes the missing valuesƒƒ
    using a K Nearest Neighbors imputer algorithm.

    Input:
        - df: dataframe
        - k: number of neighbors to use
    Output:
        - df: dataframe with missing values imputed
    '''
    # Extracting the numerical columns
    df[Omega == False] = np.nan
    numeric_df = df.drop(labels=keepcols, axis=1)
    everything_else = df[keepcols]

    # Iterating over each column
    for j in range(len(numeric_df.columns)):
        lst_missing = numeric_df.iloc[:,j][numeric_df.iloc[:, j].isnull()].index.to_list()

        # Iterating over missing rows
        for i in lst_missing:
            # Get the comparison row
            euclidean_row = numeric_df.iloc[i]
            # Take euclidean distances of the other columns
            euclidean_distances = nan_euclidean_distances(
                numeric_df, [euclidean_row.to_list()])

            # Flatten the euclidean distance array
            euc_flattened = euclidean_distances.flatten(order='F')

            # Get the k smallest distances
            k_nearest = kSmallest(euc_flattened, k)
            k_nearest_indices = np.where(np.in1d(euc_flattened, k_nearest))[0]

            # Get mean of the k nearest neighbors
            k_mean = numeric_df.iloc[k_nearest_indices].iloc[:, j].mean(skipna=True)
            
            # Assigning the value to the nan row cell
            numeric_df.iloc[i].iloc[j] = k_mean

    return pd.concat([numeric_df,everything_else], axis=1).fillna(0)
