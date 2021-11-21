import numpy as np
import pandas as pd
import math

def create_matrix(X,fractionObserved=0.9):
    '''
    Recieves a matrix X and returns a matrix with the same dimensions and
    with a fraction of the observed records.
    '''
    Omega = np.array(np.random.rand(X.shape[0],X.shape[1])< fractionObserved)
    Xobs = Omega*X
    return Xobs, Omega

