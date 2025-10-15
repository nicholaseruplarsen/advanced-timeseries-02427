# regsmooth2d.py
# adapted from matlab/regsmooth2D.m

import numpy as np
import matplotlib.pyplot as plt

def regsmooth2d(data, points, order, h, bound=None, conditional_on_y=0):
    """
    Locally weighted polynomial regression 2D
    Tri-cube weight function
    
    [x, y, z_tilde, dist_max] = regsmooth2d(data, points, order, h, bound, conditional_on_y)
    data: [x, y, z] shape (n, 3)
    bound: [minX maxX minY maxY] optional
    conditional_on_y: 0 local poly, 1 conditional parametric (global linear in y)
    """
    n = len(data)
    NN = round(h * n)
    
    if bound is None:
        min_x, max_x = np.min(data[:, 0]), np.max(data[:, 0])
        min_y, max_y = np.min(data[:, 1]), np.max(data[:, 1])
    else:
        min_x, max_x, min_y, max_y = bound
    
    scale_x = max_x - min_x
    scale_y = max_y - min_y
    data_sc = np.column_stack((
        (data[:, 0] - min_x) / scale_x * 2 - 1,
        (data[:, 1] - min_y) / scale_y * 2 - 1
    ))
    
    x_vec = np.linspace(-1, 1, points)
    y_vec = np.linspace(-1, 1, points)
    X, Y = np.meshgrid(x_vec, y_vec)
    x_grid = X.flatten()
    y_grid = Y.flatten()
    
    z_tilde = np.zeros(points ** 2)
    dist_max = np.zeros(points ** 2)
    
    for p in range(points ** 2):
        if conditional_on_y == 0:
            dist = np.sqrt((data_sc[:, 0] - x_grid[p]) ** 2 + (data_sc[:, 1] - y_grid[p]) ** 2)
        else:
            dist = np.abs(data_sc[:, 0] - x_grid[p])
        
        sort_idx = np.argsort(dist)[:NN]
        dist_nn = dist[sort_idx]
        dist_max[p] = np.max(dist_nn)
        x_nn = data_sc[sort_idx, 0]
        y_nn = data_sc[sort_idx, 1]
        z_nn = data[sort_idx, 2]
        
        # Design matrix U and ep
        U = np.ones((NN, 1))
        ep = np.array([1])
        if order == 1:
            U = np.column_stack((U, x_nn, y_nn))
            ep = np.append(ep, [x_grid[p], y_grid[p]])
        elif order == 2:
            if conditional_on_y == 0:
                U = np.column_stack((U, x_nn, y_nn, x_nn**2, y_nn**2, x_nn * y_nn))
                ep = np.append(ep, [x_grid[p], y_grid[p], x_grid[p]**2, y_grid[p]**2, x_grid[p] * y_grid[p]])
            else:
                U = np.column_stack((U, x_nn, y_nn, x_nn**2))
                ep = np.append(ep, [x_grid[p], y_grid[p], x_grid[p]**2])
        
        # Tri-cube weights
        dist_nn_norm = dist_nn / np.max(dist_nn)
        W = np.diag((1 - dist_nn_norm ** 3) ** 3)
        
        # Solve
        theta = np.linalg.solve(U.T @ W @ U, U.T @ W @ z_nn)
        
        z_tilde[p] = ep @ theta
    
    # Reshape and unscale
    x = np.linspace(min_x, max_x, points)
    y = np.linspace(min_y, max_y, points)
    X, Y = np.meshgrid(x, y)
    z_tilde = z_tilde.reshape(points, points)
    dist_max = dist_max.reshape(points, points)
    
    return X, Y, z_tilde, dist_max

# Example usage
if __name__ == "__main__":
    # Sample data: bilinear surface + noise
    np.random.seed(42)
    n = 100
    x_s = np.random.uniform(0, 10, n)
    y_s = np.random.uniform(0, 10, n)
    z_s = x_s * y_s / 100 + np.random.normal(0, 1, n)
    data = np.column_stack((x_s, y_s, z_s))
    
    X, Y, Z, _ = regsmooth2d(data, 20, 1, 0.3)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=10)
    plt.show()