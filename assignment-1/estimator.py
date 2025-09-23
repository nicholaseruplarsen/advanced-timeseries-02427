# estimator.py
# adapted from matlab/estimator.m

import numpy as np

def estimator(data, interval, order, points, smooth):
    """
    Estimator for local polynomial regression.
    
    Parameters:
    - data: array-like, the response variable
    - interval: [min, max] for the grid
    - order: polynomial order
    - points: number of grid points
    - smooth: precomputed smooth matrix (as from smooth_localreg equivalent)
    
    Returns:
    - fhat: estimated values
    """
    n = len(data)
    delta = (interval[1] - interval[0]) / (points - 1)
    index = np.floor((data - interval[0]) / delta).astype(int) + 1
    coef = np.zeros((n, order + 2))
    x_lower = np.zeros(n)
    
    for k in range(n):
        x_lower[k] = smooth[index[k] - 1, 1]  # 0-based indexing
        coef[k, order + 1] = smooth[index[k] - 1, 2]
        coef[k, 0] = smooth[index[k] - 1, 0]
        for i in range(1, order + 1):
            coef[k, i] = smooth[index[k] - 1, 3 + i - 1]
    
    dist = data - x_lower
    fhat = coef[:, 0].copy()
    for i in range(order + 1):
        fhat += coef[:, i + 1] * (dist ** i)
    
    return fhat