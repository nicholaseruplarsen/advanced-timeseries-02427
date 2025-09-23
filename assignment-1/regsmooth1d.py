# regsmooth1d.py
# adapted from matlab/regsmooth1D.m

import numpy as np
import matplotlib.pyplot as plt

def regsmooth1d(data, points, order, h):
    """
    Locally weighted polynomial regression 1D
    Tri-cube weight function
    
    [x, y_tilde] = regsmooth1d(data, points, order, h)
    data: [x, y] shape (n, 2)
    """
    n = len(data)
    NN = round(h * n)
    
    min_x = np.min(data[:, 0])
    max_x = np.max(data[:, 0])
    scale_x = max_x - min_x
    data_sc = ((data[:, 0] - min_x) / scale_x * 2 - 1).reshape(-1, 1)
    
    x_vec = np.linspace(-1, 1, points)
    x = x_vec.copy()
    y_tilde = np.zeros(points)
    
    for p in range(points):
        dist = np.abs(data_sc[:, 0] - x[p])
        sort_idx = np.argsort(dist)[:NN]
        dist_nn = dist[sort_idx]
        x_nn = data_sc[sort_idx, 0]
        z_nn = data[sort_idx, 1]
        
        # Design matrix U and eval point ep
        U = np.ones((NN, 1))
        ep = np.array([1])
        for i in range(1, order + 1):
            U = np.column_stack((U, x_nn ** i))
            ep = np.append(ep, x[p] ** i)
        
        # Tri-cube weights
        dist_nn_norm = dist_nn / np.max(dist_nn)
        W = np.diag((1 - dist_nn_norm ** 3) ** 3)
        
        # Solve normal equations
        theta = np.linalg.solve(U.T @ W @ U, U.T @ W @ z_nn)
        
        y_tilde[p] = ep @ theta
    
    # Unscale x
    x = np.linspace(min_x, max_x, points)
    
    return x, y_tilde

# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    n = 100
    x_sample = np.linspace(0, 10, n)
    y_sample = np.sin(x_sample) + np.random.normal(0, 0.1, n)
    data = np.column_stack((x_sample, y_sample))
    
    x_fit, y_fit = regsmooth1d(data, 50, 1, 0.3)
    plt.plot(data[:, 0], data[:, 1], 'o', label='Data')
    plt.plot(x_fit, y_fit, '-', label='Fit')
    plt.legend()
    plt.show()