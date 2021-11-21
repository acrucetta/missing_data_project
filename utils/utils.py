import numpy as np
import pandas as pd
import math

## read in data files


def create_matrix(X, seed=60615, fractionObserved=0.9):
    '''
    Recieves a matrix X and returns a matrix with the same dimensions and
    with a fraction of the observed records.
    '''
    np.random.seed(seed)
    Omega = np.array(np.random.rand(X.shape[0], X.shape[1]) < fractionObserved)
    Xobs = Omega * X
    return Xobs, Omega

<<<<<<< HEAD
=======

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
>>>>>>> fecac809de49d5a95926290519d8f730ad73ebd3
