# src/utils/ekf.py

import numpy as np

def ekf_parameter_estimation(y, a_init=0.5, a_var_init=1.0, sigma_v2=1.0, sigma_e2=1.0):
    """
    Extended Kalman Filter for estimating parameter a in the model:
    x_{t+1} = a x_t + v_t, v_t ~ N(0, sigma_v^2)
    y_t = x_t + e_t, e_t ~ N(0, sigma_e^2)
    
    By augmenting state z_t = [x_t, a]^T
    """
    n = len(y)
    zt = np.array([0.0, a_init])
    Pt = np.array([[sigma_e2, 0.0], [0.0, a_var_init]])
    Rv = np.array([[sigma_v2, 0.0], [0.0, 0.0]])
    Re = sigma_e2
    Ht = np.array([1.0, 0.0])
    
    z = np.zeros((n, 2))
    z[0] = zt
    avar = np.zeros(n)
    avar[0] = Pt[1,1]
    
    for i in range(n-1):
        ft = np.array([zt[1] * zt[0], zt[1]])
        Ft = np.array([[zt[1], zt[0]], [0.0, 1.0]])
        
        denom = np.dot(np.dot(Ht, Pt), Ht.T) + Re
        Kt = np.dot(np.dot(Ft, Pt), Ht.T) / denom
        
        zt = ft + Kt * (y[i] - zt[0])
        Pt = np.dot(np.dot(Ft, Pt), Ft.T) + Rv - np.outer(Kt, Kt) * (Re + np.dot(np.dot(Ht, Pt), Ht.T))
        
        z[i+1] = zt
        avar[i+1] = Pt[1,1]
    
    return z, avar