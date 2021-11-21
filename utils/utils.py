import numpy as np
import pandas as pd
import math

## read in data files


def create_matrix(X, seed=60615, fractionObserved=0.9, keepcols=None):
    '''
    Recieves a matrix X and returns a matrix with the same dimensions and
    with a fraction of the observed records.

    Inputs:
        X (Pandas Dataframe): Matrix to have random data removed
        fractionObserved (float between 0 and 1): percentage of data to keep
        keepcols (list of strings): Columns to not have any data removed
    Returns 
        X with removed data and 
        Omega, a matrix of the same size of X denoting "True" if value observed 
        or "False" otherwise
    '''
    if keepcols is not None:
        keeps = X[keepcols]
        X = X.drop(labels=keepcols, axis=1)
        omegatrue = np.ones(keeps.shape, dtype=bool)

    rand_cols = list(X.columns)
    np.random.seed(seed)
    Omega = np.array(np.random.rand(X.shape[0], X.shape[1]) < fractionObserved)

    if keepcols is not None:
        Omega = np.concatenate((Omega, omegatrue), axis=1)
        X = np.concatenate((X, keeps), axis=1)
        rand_cols = rand_cols + keepcols

    Xobs = pd.DataFrame(Omega * X, columns=rand_cols)
    return Xobs, Omega


## linear regression  (include timer)

## performance metric
def calculate_errors(actual, predict):
    '''
    Calculates the MSE and RMSE of a predicted indicator
    (actual and predict must be of same dimension)
    Inputs:
        actual (Array): array of size (n,)
        predict (Array): array of size (n,)
    Returns (tuple): MSE (Float), RMSE (Float)
    '''
    mse = np.square(np.subtract(actual, predict)).mean()
    rmse = np.sqrt(mse)
    return mse, rmse
