import numpy as np
import pandas as pd
from scipy.sparse import data

bike_file_path = "../data/Bike-Sharing-Dataset/hour.csv"
loan_file_path = "../data/loan-default-data/Training Data.csv"

def read_in(file_path,  dataset):
    '''
    reads in a filepath (specifically for bike and loans) and returns
    DataFrame of y data and X data

    Inputs:
        file_path (str): path to find file
        dataset (str): name of dataset
    Returns tuple of y (DataFrame) and X (DataFrame)
    '''
    df = pd.read_csv(file_path)
    if dataset is "Bike":
        y = df["cnt"]
        X = df.drop(["cnt"], axis = 1)
    if dataset is "Loan":
        y = df["Risk_Flag"]
        X = df.drop(["Risk_Flag"], axis = 1)
    y = pd.DataFrame(y)
    return y, X


def create_matrix(Xorg, seed=60615, fractionObserved=0.9, keepcols=None):
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
        keeps = Xorg[keepcols]
        X = Xorg.drop(labels=keepcols, axis=1)
        omegatrue = np.ones(keeps.shape, dtype=bool)
    else:
        X = Xorg
    rand_cols = list(X.columns)
    np.random.seed(seed)
    Omega = np.array(np.random.rand(X.shape[0], X.shape[1]) < fractionObserved)

    if keepcols is not None:
        Omega = np.concatenate((Omega, omegatrue), axis=1)
        X = pd.concat((X.astype(float), keeps), axis=1)
        rand_cols = rand_cols + keepcols
    Xobs = pd.DataFrame(Omega * X, columns=rand_cols)
    return Xobs, Omega


## performance metric
def calculate_errors(actual, predict, continuous=True):
    '''
    Calculates error metrics of a predicted indicator
    (actual and predict must be of same dimension)
    Inputs:
        actual (Array): array of size (n,)
        predict (Array): array of size (n,)
        continuous (Bool): whether 
    Returns (tuple) of 
        two error metrics (MSE/RMSE if continuous, accuracy/F1 score otherwise)
    '''
    if continuous:
        error1 = np.square(np.subtract(actual, predict)).mean()
        error2 = np.sqrt(error1)
    else:
        error1 = (actual == predict).sum() / actual.shape[0]
        truepositive = np.sum((predict == 1) & (actual == 1))
        precision = truepositive / np.sum(predict == 1)
        recall = truepositive / (truepositive + np.sum((predict == 0)  & 
                                                       (actual == 1)))
        error2 = 2 * precision * recall /(precision + recall)
    return (error1, error2)
