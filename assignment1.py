import numpy as np 
import matplotlib.pyplot as plt 
from numpy.linalg import qr
from numpy.linalg import cholesky
 

def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Generate polynomial feature matrix with powers from 0 to degree.
    
    Parameters
    ----------
    x : np.ndarray
        Input data vector of shape (n_samples,)
    degree : int
        Maximum polynomial degree
        
    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_samples, degree + 1) where A[i, j] = x[i]^j
    """
    n = len(x)
    A = np.zeros((n, degree + 1))
    
    for j in range(degree + 1):
        A[:, j] = x**j
    
    return A