import numpy as np
from scipy.optimize import least_squares
from statsmodels.tsa.tsatools import lagmat
from statsmodels.api import add_constant
from statsmodels.regression.linear_model import OLS

class STAR:
    def __init__(self, p, d):
        self.p = p
        self.d = d

    def fit(self, data):
        n = len(data)
        max_lag = max(self.p, self.d)
        endog = data[max_lag:]
        lags = add_constant(lagmat(data, self.p)[max_lag:, :])
        threshold_var = data[max_lag - self.d : n - self.d]
        def residual(params):
            a = params[0:self.p+1]
            b = params[self.p+1:2*(self.p+1)]
            gamma = params[-2]
            c = params[-1]
            I = 1 / (1 + np.exp(-gamma * (threshold_var - c)))
            pred = lags @ a + (lags @ b) * I
            return endog - pred
        lin_model = OLS(endog, lags).fit()
        a_init = lin_model.params
        b_init = np.zeros(self.p + 1)
        gamma_init = 10.0
        c_init = np.median(threshold_var)
        params_init = np.concatenate((a_init, b_init, [gamma_init, c_init]))
        res = least_squares(residual, params_init)
        self.params = res.x
        self.fittedvalues = np.full(n, np.nan)
        self.fittedvalues[max_lag:] = endog - res.fun
        self.resid = data - self.fittedvalues
        k = len(self.params)
        self.ssr = np.sum(res.fun ** 2)
        self.aic = n * np.log(self.ssr / (n - max_lag)) + 2 * k