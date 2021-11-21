import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import nan_euclidean_distances

'''
To find out the weights following steps have to be taken:

1) Choose missing value to fill in the data.
2) Select the values in a row
3) Choose the number of neighbors you want to work with (ideally 2-5)
4)Calculate Euclidean distance from all other data points corresponding to each other in the row.
5) Select the smallest 2 and average out.

Sources used:
- https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/
- https://www.numpyninja.com/post/mice-and-knn-missing-value-imputations-through-python
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html
- https://chrisalbon.com/code/machine_learning/preprocessing_structured_data/imputing_missing_class_labels_using_k-nearest_neighbors/

'''

def normalize_data(df):
    '''
    Normalizes the data in X
    '''
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    return df


def calculate_euclidean_distance(a, b):
    '''
    Calculates euclidean distance between point x
    and point y.
    '''
    temp = a-b
    dist = np.sqrt(np.dot(temp.T, temp))
    return dist

def knn_classifier(X, y, k=5):
    '''
    Recieves a matrix X and a vector y and uses a KNN
    classifier to predict the class of each point in X.
    '''
    pass
    
def KNN_imputation(X, k=5):
    '''
    Recieves a matrix X and imputes the missing values
    using a K Nearest Neighbors imputer.
    '''
    pass