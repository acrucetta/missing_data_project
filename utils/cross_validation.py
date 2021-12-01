import random
import pandas as pd
import numpy as np

seed = 4358

def cross_validation(X, y, seed):
    
    error_calc = []
    
    for i in range(30):
        random.seed(seed)

        Omega = missing_obs(seed, X)
        impute_mean_X = impute_mean(Omega, X)
        deletion_y, deletion_X = delete_missing(Omega, y, X)
        svt_X = singular_value_thresholding(Omega, X)
        KNN_X = KNN(Omega, X)
        EM_X = EM(Omega, X)

        seed = seed/i
