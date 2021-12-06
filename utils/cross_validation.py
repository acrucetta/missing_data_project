import random
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

def cross_validation(X, y, seed, keepcols=None, obsp=.9):
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
    error_calc = {"mean_X":{"er1":[], "er2":[]}, 
#                  "deletion":{"er1":[], "er2":[]}, 
                  "svt_X": {"er1":[], "er2":[]},
                  "knn_X": {"er1":[], "er2":[]}, 
                  "em_X": {"er1":[], "er2":[]}}
    for i in range(iters):
        random.seed(seed)
        Xobs, Omega = utils.create_matrix(X, seed=seed, 
                                    fractionObserved=obsp, keepcols=keepcols)
        mean_X = im.impute_mean(Omega, Xobs)
        svt_X = im.singular_value_thresholding(Omega, Xobs, keepcols)
        knn_X = ki.KNN_imputation(Xobs, Omega, keepcols)
        knn_X = knn_X.fillna(0)
        em_X = em.expecation_maximization(Xobs, Omega, keepcols)
        for i, matrix in enumerate([mean_X, svt_X, knn_X, em_X['X_imputed']]):
            matrix_oh = pd.get_dummies(matrix)
            yhat = reg.predict_least_squares(matrix_oh, y)
            er1, er2 = utils.calculate_errors(y, yhat)
            error_calc[RECODE[i]]["er1"] += [er1]
            error_calc[RECODE[i]]["er2"] += [er2]
        # deletion_y, deletion_X = im.delete_missing(Omega, y, Xobs)
        # del_X_oh = pd.get_dummies(deletion_X)
        # yhat_del = reg.predict_least_squares(del_X_oh, deletion_y)
        # error_calc["deletion"] += utils.calculate_errors(y_deletion, yhat_det)
        seed = (seed // (i + 1)) + (i ** 2)
    if type(error_calc["mean_X"]["er1"][0]) in [float, int]:
        error_means = {idx: {key: mean(idx) for key, idx in j.items()}
                       for idx, j in error_calc.items()}     
    else:
        error_means = {idx: {key: pd.concat(idx).mean() for key, idx in j.items()}
                       for idx, j in error_calc.items()}
    return error_means
