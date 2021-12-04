import random
import pandas as pd
import numpy as np
import utils
import impute_methods as im
import expectation_maximization as em
import knn_imputation as ki
import regression_func as reg

seed = 4358

def cross_validation(X, y, seed, keepcols=None, obsp=.9):
    
    error_calc = {"mean_X":[], "deletion":[], "svt_X": [], "knn_X": [], "em_X": []}
    
    for i in range(30):
        random.seed(seed)
        Xobs, Omega = utils.create_matrix(X, seed=seed, 
                                    fractionObserved=obsp, keepcols=keepcols)
        mean_X = im.impute_mean(Omega, Xobs)
        svt_X = im.singular_value_thresholding(Omega, Xobs, keepcols)
        knn_X = ki.KNN_imputation(Xobs, Omega, keepcols)
        em_X = em.expecation_maximization(Xobs, Omega, keepcols)
        for matrix in [mean_X, svt_X, knn_X, em_X]:
            yhat = reg.predict_least_squares(matrix, y)
            error_calc[matrix] += utils.calculate_errors(y, yhat)
        deletion_y, deletion_X = im.delete_missing(Omega, y, Xobs)
        yhat_del = reg.predict_least_squares(deletion_X, deletion_y)
        error_calc["deletion"] += utils.calculate_errors(y_deletion, yhat_det)

        seed = (seed // i) + (i ** 2)
    
    return error_cal
