# smooth_localreg.py
# adapted from matlab/smooth_localreg.m

import numpy as np
from scipy import sparse

def smooth_localreg(data, interval, order, points, h):
    """
    Smoother using local polynomial regression.
    
    Usage: coef = smooth_localreg(data, interval, order, points, h)
    data: two column array, data[:,0] = f(data[:,1])
    """
    n = len(data)
    coef = np.zeros((points, 4 + order))  # [tilde_y, x, c, coefs]
    x = np.linspace(interval[0], interval[1], points)
    coef[:, 1] = x
    
    U_p = np.ones((points, order + 1))
    for j in range(1, order + 1):
        U_p[:, j] = x ** j
    
    for i in range(points):
        w = np.abs(data[:, 1] - x[i]) / h
        w = (1 - w ** 3) ** 3
        w = (w + np.abs(w)) / 2  # Ensure non-negative
        # W = sparse.diags(w, 0, n, n).toarray()  # Dense for simplicity
        W = sparse.diags(w, 0, shape=(n, n)).toarray()  # Dense for simplicity
        
        U = np.ones((n, order + 1))
        for j in range(1, order + 1):
            U[:, j] = (data[:, 1] - x[i]) ** j
        
        theta = np.linalg.solve(U.T @ W @ U, U.T @ W @ data[:, 0])
        coef[i, 3:3+order+1] = theta
        coef[i, 0] = theta[0]  # Constant term
    
    # Derivative approximation (as in MATLAB)
    deltax = x[1] - x[0]
    nom = coef[1:, 0] - coef[:-1, 0]
    for i in range(1, order + 1):
        nom -= coef[:-1, 3 + i] * (deltax ** i)
    coef[:-1, 2] = nom / (deltax ** (order + 1))
    
    return coef