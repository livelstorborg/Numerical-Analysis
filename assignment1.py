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




def forward(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve lower triangular system Ax = b using forward substitution.
    
    Parameters
    ----------
    A : np.ndarray
        Lower triangular coefficient matrix of shape (n, n)
    b : np.ndarray
        Right-hand side vector of shape (n,)
        
    Returns
    -------
    np.ndarray
        Solution vector x of shape (n,)
    """
    n = A.shape[0]
    if b.shape[0] != n:
        raise ValueError('Input dimensions do not match')
    
    x = np.zeros(n)
    
    for k in range(n):  
        if abs(A[k, k]) > 1e-12:
            temp = 0
            for j in range(k):  
                temp += A[k, j] * x[j]
            x[k] = (b[k] - temp) / A[k, k]
        else:
            raise ValueError('Input singular')
    
    return x