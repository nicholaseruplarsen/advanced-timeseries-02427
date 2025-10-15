# Corrected EKF implementation to be used in Part 4b
import numpy as np

def ekf_parameter_estimation_corrected(y, a_init=0.5, a_var_init=1.0, sigma_v2=1.0, sigma_e2=1.0):
    """
    Corrected Extended Kalman Filter for estimating parameter 'a' in the model:
    x_{t+1} = a * x_t + v_t,  v_t ~ N(0, sigma_v2)
    y_t = x_t + e_t,      e_t ~ N(0, sigma_e2)

    By augmenting the state z_t = [x_t, a_t]^T.
    """
    n = len(y)
    # Augmented state z_t = [x_t, a_t]^T
    # Noise covariances
    Rv = np.array([[sigma_v2, 0.0], [0.0, 0.0]])  # Process noise for [x, a]
    Re = sigma_e2  # Measurement noise

    # Observation matrix H_t = [1, 0]
    Ht = np.array([[1.0, 0.0]])

    # Initialize state and covariance at t=0. This is z_{0|0}, P_{0|0}
    # A reasonable initial guess for the state x_0 is the first observation y_0.
    z_est = np.array([y[0], a_init])
    P_est = np.array([[sigma_e2, 0.0], [0.0, a_var_init]])

    # Store history of estimates
    z_history = np.zeros((n, 2))
    avar_history = np.zeros(n)
    z_history[0] = z_est
    avar_history[0] = P_est[1, 1]

    # EKF loop for t = 1, ..., n-1
    for t in range(1, n):
        # --- PREDICTION STEP (from t-1 to t) ---
        # Predicted state z_{t|t-1} = f(z_{t-1|t-1})
        z_pred = np.array([z_est[1] * z_est[0], z_est[1]])

        # Jacobian of transition function f, evaluated at z_{t-1|t-1}
        Ft = np.array([[z_est[1], z_est[0]], [0.0, 1.0]])

        # Predicted covariance P_{t|t-1}
        P_pred = Ft @ P_est @ Ft.T + Rv

        # --- UPDATE STEP (at time t, using measurement y[t]) ---
        # Innovation (prediction error)
        innovation = y[t] - (Ht @ z_pred)

        # Innovation covariance
        S = Ht @ P_pred @ Ht.T + Re

        # Kalman Gain
        K = (P_pred @ Ht.T) / S

        # Updated state estimate z_{t|t}
        z_est = z_pred + K.flatten() * innovation.flatten()

        # Updated covariance P_{t|t} (Joseph form for numerical stability)
        I_KH = np.eye(2) - K @ Ht
        P_est = I_KH @ P_pred @ I_KH.T + K @ np.array([[Re]]) @ K.T
        
        # Store results for time t
        z_history[t] = z_est
        avar_history[t] = P_est[1, 1]

    return z_history, avar_history