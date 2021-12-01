import numpy as np
import pandas as pd

def impute_mean(Omega, X):
    '''
    Take the data with missing values and impute with the column mean
    '''
    new = X*Omega
    means = new.mean(axis=0)
    X = X.mask(df*(1-Omega)==1).fillna(means)
    return X


def delete_missing(Omega, y, X):
    '''
    Drop the rows with missing values in both the X and y datasets
    '''
    final = []
    for i in [y,X]:
        r = Omega.shape[1] - Omega.sum(axis = 1)
        i["keeps"] = r
        new = i.loc[y['keeps'] == 0]
        new = new.drop(["keeps"], axis =1)
        final.append(new)
    return final[0], final[1]


def singular_value_thresholding(Omega, X):
    '''
    Use singular value thresholding to loop through and fill the missing values
    '''
    Xobs = Omega*X
    tau = 30
    stopping_value = 0.1
    X_hat = Xobs
    X_old = np.zeros((X.shape[0],X.shape[1]))

    while np.linalg.norm(X_hat -X_old) > stopping_value:
        X_old = X_hat
        u, s, vt = np.linalg.svd(X_hat, full_matrices=False)
        st = np.where(s > tau, s, 0)
        X_new = u @ np.diag(st) @ vt
        X_hat = Xobs*Omega + X_new*(1 - Omega)

    return X