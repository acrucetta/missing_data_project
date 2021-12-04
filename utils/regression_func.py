import numpy as np
import pandas as pd
import time

'''
Creating a regression function that distinguishes
between a linear regressor or a linear classifier.
'''

def least_squares(X,y):
    '''
    Trains a least squares algorithm on the data.
    
    Inputs:
        X (Pandas Dataframe): Matrix of features
        y (Pandas Dataframe): Matrix of target
    Returns:
        w (Array): Array of weights
    '''
    # Calculate the weights
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w

def predict_least_squares(X, y):
    '''
    Predicts the target using a least squares algorithm.
    
    Inputs:
        X (Pandas Dataframe): Matrix of features
        w (Array): Array of weights
    Returns:
        y (Array): Array of target
    '''
    # Calculate the target
    w = least_squares(X, y)
    yhat = X.dot(w)
    return yhat


