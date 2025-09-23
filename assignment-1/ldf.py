# ldf.py
# adapted from ldf.R
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.utils import resample
from smooth_localreg import smooth_localreg
from estimator import estimator

def leave_one_out_equiv(D):
    # D: np.array shape (n, 2), col0: y, col1: x
    # Fix: ensure all spans are in valid range [0, 1] for lowess
    spans = np.arange(0.2, 1.0, 0.1)  # Only use valid frac values
    
    RSSk_all = []
    for span in spans:
        residuals = []
        for i in range(len(D)):
            D_train = np.delete(D, i, axis=0)
            res = lowess(D_train[:, 0], D_train[:, 1], frac=span, return_sorted=True)
            pred = np.interp(D[i, 1], res[:, 0], res[:, 1], left=np.nan, right=np.nan)
            if np.isnan(pred):
                continue  # Skip if out of range
            residuals.append(D[i, 0] - pred)
        RSSk_all.append(np.sum(np.array(residuals)**2))
    
    RSSk_all = np.array(RSSk_all)
    span_best = spans[np.argmin(RSSk_all)]
    fit_best = lowess(D[:, 0], D[:, 1], frac=span_best, return_sorted=False)
    RSSk = np.sum((D[:, 0] - fit_best)**2)
    
    return RSSk

def ldf_r_like_fixed(x, lags, frac=0.75, n_boot=30):
    # Ensure frac is in valid range
    frac = max(0.1, min(1.0, frac))
    
    val = []
    for k in lags:
        D = np.column_stack((x[k:], x[:-k]))
        fit = lowess(D[:, 0], D[:, 1], frac=frac, return_sorted=False)
        RSSk = np.sum((D[:, 0] - fit)**2)
        RSS = np.sum((D[:, 0] - np.mean(D[:, 0])) ** 2)
        val.append((RSS - RSSk) / RSS)
    
    iid_val = []
    for i in range(n_boot):
        print(f"Calculating bootstrap no. {i+1} of {n_boot}")
        xr = resample(x, n_samples=min(len(x), 100), replace=True)
        DR = np.column_stack((xr[1:], xr[:-1]))
        RSSk_r = leave_one_out_equiv(DR)
        RSS_r = np.sum((DR[:, 0] - np.mean(DR[:, 0])) ** 2)
        iid_val.append((RSS_r - RSSk_r) / RSS_r)
    
    return np.array(val), np.quantile(iid_val, 0.95)

def ldf(data, order, points, h, maxlag):
    n = len(data)
    interval = [np.min(data), np.max(data)]
    SS0 = np.sum((data[maxlag:] - np.mean(data[maxlag:])) ** 2)
    phi = np.zeros(maxlag)
    for i in range(1, maxlag + 1):
        lag_data = np.column_stack((data[maxlag:], data[maxlag - i:n - i]))
        smooth = smooth_localreg(lag_data, interval, order, points, h)
        fit = estimator(data[maxlag - i:n - i], interval, order, points, smooth)
        res = data[maxlag:] - fit
        SS0k = np.sum(res ** 2)
        R20 = (SS0 - SS0k) / SS0
        phi[i-1] = np.sign(smooth[-1, 0] - smooth[0, 0]) * np.sqrt((np.abs(R20) + R20) / 2)
    return np.concatenate(([1], phi))

def ldf_one(data, order, points, h, lag):
    interval = [np.min(data), np.max(data)]
    lag_data = np.column_stack((data[lag:], data[:-lag]))
    SS0 = np.sum((lag_data[:, 0] - np.mean(lag_data[:, 0])) ** 2)
    smooth = smooth_localreg(lag_data, interval, order, points, h)
    fit = estimator(lag_data[:, 1], interval, order, points, smooth)
    res = lag_data[:, 0] - fit
    SS0k = np.sum(res ** 2)
    R20 = (SS0 - SS0k) / SS0
    phi = np.sign(smooth[-1, 0] - smooth[0, 0]) * np.sqrt((np.abs(R20) + R20) / 2)
    
    return {
        'phi': phi,
        'lag_data': lag_data,
        'fit': fit,
        'smooth': smooth,
        'lag': lag
    }