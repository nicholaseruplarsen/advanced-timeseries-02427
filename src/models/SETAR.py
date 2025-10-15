import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat
from statsmodels.api import add_constant

class SETAR:
    def __init__(self, ar_order, num_regimes, delay, min_frac=0.15, grid_size=50):
        self.ar_order = ar_order
        self.num_regimes = num_regimes
        self.delay = delay
        self.min_frac = min_frac
        self.grid_size = grid_size

    def fit(self, data):
        n = len(data)
        nobs_initial = max(self.ar_order, self.delay)
        endog = data[nobs_initial:]
        threshold_var = data[nobs_initial - self.delay : n - self.delay]
        z_sorted = np.sort(threshold_var)
        low = z_sorted[int(len(z_sorted) * self.min_frac)]
        high = z_sorted[int(len(z_sorted) * (1 - self.min_frac))]
        threshold_grid = np.linspace(low, high, self.grid_size)
        rss_list = []
        for th in threshold_grid:
            indicators = np.searchsorted([th], threshold_var)
            indicator_matrix = (indicators[:, None] == np.arange(self.num_regimes))
            lags = add_constant(lagmat(data, self.ar_order)[nobs_initial:, :])
            exog = np.multiply(
                np.tile(lags, (1, self.num_regimes)),
                np.kron(indicator_matrix, np.ones((1, self.ar_order + 1)))
            )
            if np.any(np.sum(indicator_matrix, axis=0) < (self.ar_order + 1)):
                continue
            model = OLS(endog, exog).fit()
            rss_list.append(model.ssr)
        if rss_list:
            best_idx = np.argmin(rss_list)
            best_threshold = threshold_grid[best_idx]
            print(f"Best threshold: {best_threshold:.3f}")
            indicators = np.searchsorted([best_threshold], threshold_var)
            indicator_matrix = (indicators[:, None] == np.arange(self.num_regimes))
            lags = add_constant(lagmat(data, self.ar_order)[nobs_initial:, :])
            exog = np.multiply(
                np.tile(lags, (1, self.num_regimes)),
                np.kron(indicator_matrix, np.ones((1, self.ar_order + 1)))
            )
            self.model = OLS(endog, exog).fit()
            self.fittedvalues = np.full(n, np.nan)
            self.fittedvalues[nobs_initial:] = self.model.fittedvalues
            self.resid = np.full(n, np.nan)
            self.resid[nobs_initial:] = self.model.resid
            k = (self.ar_order + 1) * self.num_regimes + (self.num_regimes - 1)
            self.aic = n * np.log(self.model.ssr / (n - nobs_initial)) + 2 * k
        else:
            print("No valid regime")