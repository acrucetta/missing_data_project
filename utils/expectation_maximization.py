import functools
import numpy as np
import pandas as pd

def clean_toy_set(x):
    x["Numerical A"] = x["Numerical A"].fillna(5)
    x["Numerical B"] = x["Numerical B"].fillna(20)
    x["Categorical D"] = x["Categorical D"].fillna(" N ") 
    x["Categorical C"] = x["Categorical C"].fillna(" B ")
    x["Categorical C"][x["Categorical C"] == " B "] = 1
    x["Categorical C"][x["Categorical C"] == " A "] = 0
    x["Categorical D"][x["Categorical D"] == " Y "] = 1
    x["Categorical D"][x["Categorical D"] == " N "] = 0


def expecation_maximization(Xobs, Omega, keepcols, max_iter=1000, eps=.00001):
    '''
    Fills in missing values of a matrix using the expectation maximation
    algorithm to determine the maximum likelihood estimate.
    
    Input:
        X (DataFrame): matrix with missing values
        Omega (Array): matrix that show where the missing values in X are
        max_iter (int): the maximum number of iterations the loop will attempt
        eps (float): parameter to determine convergence
    
    Returns dictionary of:
        mu (Series): estimated mean of each row
        Sigma (Array): estimated covariance of the matrix
        X_imputed (DataFrame): matrix with the imputed values
        Omega (Array): The original omega matrix
        Iteration (int): Number of iterations completed

    Source: https://joon3216.github.io/research_materials/2019/em_imputation_python.html
    '''
    X = Xobs.drop(labels=keepcols, axis=1)
    everything_else = Xobs[keepcols]
    Omega = Omega[:,:-len(keepcols)]
    nr, nc = X.shape
    # Collect M_i and O_i's
    one_to_nc = np.arange(1, nc + 1, step = 1)
    M = one_to_nc * (Omega == False) - 1
    O = one_to_nc * Omega - 1
    X[Omega == False] = np.nan
    # Generate Mu_0 and Sigma_0
    Mu = np.nanmean(X, axis = 0)
    observed_rows = np.all(Omega == True, axis=1)
    S = np.cov(X[observed_rows].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis = 0))
    
    # Start updating
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i, ]) != set(one_to_nc - 1): # missing component exists
                M_i = M[i, ][M[i, ] != -1]
                O_i = O[i, ][O[i, ] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                Mu_tilde[i] = Mu[M_i] + S_MO @ np.linalg.inv(S_OO) @ \
                             (X_tilde.iloc[i, O_i] - Mu[O_i])
                X_tilde.iloc[i, M_i] = Mu_tilde[i]
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
        Mu_new = np.mean(X_tilde, axis = 0)
        S_new = np.cov(X_tilde.T, bias = 1) + \
                functools.reduce(np.add, S_tilde.values()) / nr
        no_conv = np.linalg.norm(Mu - Mu_new) >= eps or \
                  np.linalg.norm(S - S_new, ord = 2) >= eps
        Mu = Mu_new
        S = S_new
        iteration += 1
    
    result = {'mu': Mu,
              'Sigma': S,
              'X_imputed': pd.concat([X_tilde, everything_else], axis=1),
              'iteration': iteration}
    
    return result