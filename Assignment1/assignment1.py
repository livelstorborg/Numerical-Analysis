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


def backward(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve upper triangular system Ax = b using backward substitution.
    
    Parameters
    ----------
    A : np.ndarray
        Upper triangular coefficient matrix of shape (n, n)
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
    
    for k in range(n-1, -1, -1):
        if abs(A[k, k]) > 1e-12:
            temp = 0
            for j in range(k+1, n):
                temp += A[k, j] * x[j]
            x[k] = (b[k] - temp) / A[k, k]
        else:
            raise ValueError('Input singular')
    
    return x


def OLS_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve ordinary least squares using QR decomposition.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, m)
    y : np.ndarray
        Response vector of shape (n,)
        
    Returns
    -------
    np.ndarray
        Parameter estimates of shape (m,)
    """
    Q, R = qr(X)
    
    m = X.shape[1]  
    Q_thin = Q[:, :m]  
    R_thin = R[:m, :m] 
    
    y_rotated = Q_thin.T @ y
    theta = backward(R_thin, y_rotated)
    
    return theta


def OLS_cholesky(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve ordinary least squares using Cholesky factorization.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, m)
    y : np.ndarray
        Response vector of shape (n,)
        
    Returns
    -------
    np.ndarray
        Parameter estimates of shape (m,)
    """
    
    R = np.linalg.cholesky(X.T @ X).T  

    z = forward(R.T, X.T @ y)
    theta = backward(R, z)
    
    return theta




# --------------- Plotting ---------------

n = 30
start = -2
stop = 2
    
np.random.seed(1)
x = np.linspace(start, stop, n)
eps = 1
r = np.random.rand(n) * eps
    
y1 = x * (np.cos(r + 0.5 * x**3) + np.sin(0.5 * x**3))     # Dataset 1
y2 = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r   # Dataset 2 

y1_true = x * (np.cos(0.5 * x**3) + np.sin(0.5 * x**3))
y2_true = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10  

A3 = polynomial_features(x, 2)  # Degree 2 polynomial (3 coefficients)
A8 = polynomial_features(x, 7)  # Degree 7 polynomial (8 coefficients)




# --------------- m=3 (polynomial degree 2) ---------------

# ----- Dataset 1 -----
theta1_qr_m3 = OLS_qr(A3, y1)
theta1_chol_m3 = OLS_cholesky(A3, y1)
y1_qr_m3 = A3 @ theta1_qr_m3
y1_chol_m3 = A3 @ theta1_chol_m3


plt.figure(figsize=(8, 5))
plt.scatter(x, y1, color='white', edgecolors='steelblue', s=50, label='Data')
plt.plot(x, y1_qr_m3, '-', color='blue', linewidth=2, label='QR', alpha=0.8)
plt.plot(x, y1_chol_m3, '--', color='red', linewidth=1.5, label='Cholesky', alpha=1)
plt.legend(fontsize=16)
# plt.title(f'Degree 2 polynomial fit of ' + r'$y = x (\cos{\frac{x^3}{2}} + \sin{\frac{x^3}{2}})$')
plt.xlabel('x', fontsize=16)
plt.ylabel('Data values ' + r'$y(x)$', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('dataset1_degree2.pdf')
plt.show()

# ----- Dataset 2 -----
theta2_qr_m3 = OLS_qr(A3, y2)
theta2_chol_m3 = OLS_cholesky(A3, y2)
y2_qr_m3 = A3 @ theta2_qr_m3
y2_chol_m3 = A3 @ theta2_chol_m3

plt.figure(figsize=(8, 5))
plt.scatter(x, y2, color='white', edgecolors='steelblue', s=50, label='y')
# plt.plot(x, y2_true, label='True y')
plt.plot(x, y2_qr_m3, '-', color='blue', linewidth=2, label='QR', alpha=0.8)
plt.plot(x, y2_chol_m3, '--', color='red', linewidth=1.5, label='Cholesky', alpha=1)
plt.legend(fontsize=16)
# plt.title(f'Degree 2 polnomial fit of ' + r'$y = 4x^5 - 5x^4 -20x^3 + 10x^2 + 40x + 10$')
plt.xlabel('x', fontsize=16)
plt.ylabel('Data values ' + r'$y(x)$', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('dataset2_degree2.pdf')
plt.show()




# --------------- m=8 (polynomial degree 7) ---------------

# ----- Dataset 1 -----
theta1_qr_m8 = OLS_qr(A8, y1)
theta1_chol_m8 = OLS_cholesky(A8, y1)
y1_qr_m8 = A8 @ theta1_qr_m8
y1_chol_m8 = A8 @ theta1_chol_m8

plt.figure(figsize=(8, 5))
plt.scatter(x, y1, color='white', edgecolors='steelblue', s=50, label='y')
# plt.plot(x, y1_true, label='True y')
plt.plot(x, y1_qr_m8, '-', color='blue', linewidth=2, label='QR', alpha=0.8)
plt.plot(x, y1_chol_m8, '--', color='red', linewidth=1.5, label='Cholesky', alpha=1)
plt.legend(fontsize=16)
# plt.title(f'Degree 7 polynomial fit of ' + r'$y = x (\cos{\frac{x^3}{2}} + \sin{\frac{x^3}{2}})$')
plt.xlabel('x', fontsize=16)
plt.ylabel('Data values ' + r'$y(x)$', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('dataset1_degree7.pdf')
plt.show()

# ----- Dataset 2 -----
theta2_qr_m8 = OLS_qr(A8, y2)
theta2_chol_m8 = OLS_cholesky(A8, y2)
y2_qr_m8 = A8 @ theta2_qr_m8
y2_chol_m8 = A8 @ theta2_chol_m8

plt.figure(figsize=(8, 5))
plt.scatter(x, y2, color='white', edgecolors='steelblue', s=50, label='y')
# plt.plot(x, y2_true, label='True y')
plt.plot(x, y2_qr_m8, '-', color='blue', linewidth=2, label='QR', alpha=0.8)
plt.plot(x, y2_chol_m8, '--', color='red', linewidth=1.5, label='Cholesky', alpha=1)
plt.legend(fontsize=16)
plt.xlabel('x', fontsize=16)
plt.ylabel('Data values ' + r'$y(x)$', fontsize=16)
# plt.title(f'Degree 7 polnomial fit of ' + r'$y = 4x^5 - 5x^4 -20x^3 + 10x^2 + 40x + 10$')
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('dataset2_degree7.pdf')
plt.show()