import random
import time
import pandas as pd
import numpy as np
import utils
import impute_methods as im
import expectation_maximization as em
import knn_imputation as ki
import regression_func as reg
from statistics import mean

seed = 4358
RECODE = {0: "mean_X", 1: "svt_X", 2: "knn_X", 3: "em_X"}

#TODO:
# - Discuss whether we're going to iterate over the different parameters to cross-validate. 
# (i.e. if we're going to do this for each of the different imputation methods)

def cross_validation(X, y, seed, tau, keepcols=None, obsp=.9, iters=30,verbose=True):
    '''
    Input:
        X: pandas dataframe
        y: pandas series
        seed: int
        keepcols: list of strings
        obsp: float
    Output:
        error_means: list of errors
    '''
    error_calc = {"mean_X":{"er1":[], "er2":[], "time":[]}, 
                  "deletion":{"er1":[], "er2":[], "time":[]},
                  "svt_X": {"er1":[], "er2":[], "time":[]},
                  "knn_X": {"er1":[], "er2":[], "time":[]}, 
                  "em_X": {"er1":[], "er2":[], "time":[]}}
    start = time.time()
    for i in range(iters):
        random.seed(seed)
        Xobs, Omega = utils.create_matrix(X, seed=seed, 
                                    fractionObserved=obsp, keepcols=keepcols)
        mean_start = time.time()
        if verbose:
            print(f"At mean, iteration {i}, time: {mean_start}")
        mean_X = im.impute_mean(Omega, Xobs)
        svt_start = time.time()
        error_calc["mean_X"]["time"] += [svt_start - mean_start]
        if verbose:
            print(f"At SVT, iteration {i}, time: {svt_start}")
        svt_X = im.singular_value_thresholding(Omega, Xobs, tau, keepcols)
        knn_start = time.time()
        error_calc["svt_X"]["time"] += [knn_start - svt_start]
        if verbose:
            print(f"At KNN, iteration {i}, time: {knn_start}")
        knn_X = ki.KNN_imputation(Xobs, Omega, keepcols)
        em_start = time.time()
        error_calc["knn_X"]["time"] += [em_start - knn_start]
        if verbose:
            print(f"At EM, iteration {i}, time: {em_start}")
        em_X = em.expecation_maximization(Xobs, Omega, keepcols)
        ls_start = time.time()
        error_calc["em_X"]["time"] += [ls_start - em_start]
        if verbose:
            print(f"At LS, iteration {i}, time: {ls_start}")
        for m, matrix in enumerate([mean_X, svt_X, knn_X, em_X['X_imputed']]):
            matrix_oh = pd.get_dummies(matrix)
            matrix_oh = matrix_oh.drop(labels=["keeps"], axis=1, errors='ignore')
            yhat = reg.predict_least_squares(matrix_oh, y)
            er1, er2 = utils.calculate_errors(y, yhat)
            error_calc[RECODE[m]]["er1"] += [er1['cnt']]
            error_calc[RECODE[m]]["er2"] += [er2['cnt']]
        del_start = time.time()
        if verbose:
            print(f"At Deletion, iteration {i}, time: {del_start}")
        deletion_y, deletion_X = im.delete_missing(Omega, y, Xobs)
        error_calc["deletion"]["time"] += [time.time() - del_start]
        del_X_oh = pd.get_dummies(deletion_X)
        yhat_del = reg.predict_least_squares(del_X_oh, deletion_y)
        er1, er2 = utils.calculate_errors(deletion_y, yhat_del)
        error_calc["deletion"]["er1"] += [er1['cnt']]
        error_calc["deletion"]["er2"] += [er2['cnt']]     
        seed = (seed // (i + 1)) + (i ** 2)
        error_means = {idx: {key: mean(idx) for key, idx in j.items()}
                       for idx, j in error_calc.items()}     
    return error_means

