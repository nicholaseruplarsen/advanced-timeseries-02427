import numpy as np
from statsmodels.tsa.tsatools import lagmat
from statsmodels.api import add_constant, WLS
from scipy.stats import norm

class IGAR:
    def __init__(self, order=2, n_regimes=2, max_iter=100, tol=1e-4):
        self.order = order
        self.n_regimes = n_regimes
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, data):
        n = len(data)
        T = n - self.order
        if T <= 0:
            raise ValueError("Data length too short for the order")
        Y = data[self.order:]
        X = lagmat(data, self.order, trim='both')  # T x order

        # Initial parameters: random assignment
        rng = np.random.RandomState(42)
        labels = rng.choice(self.n_regimes, T)
        pi = np.bincount(labels, minlength=self.n_regimes) / T
        phi = np.zeros((self.n_regimes, self.order + 1))
        sigma2 = np.zeros(self.n_regimes)
        for j in range(self.n_regimes):
            mask = labels == j
            if np.sum(mask) < self.order + 1:
                # Fallback: use overall
                exog = add_constant(X)
                model = WLS(Y, exog, weights=np.ones(T)).fit()
                phi[j] = model.params
                sigma2[j] = np.mean((Y - model.fittedvalues)**2)
                continue
            Xj = X[mask]
            Yj = Y[mask]
            exog_j = add_constant(Xj)
            model_j = WLS(Yj, exog_j, weights=np.ones(len(Yj))).fit()
            phi[j] = model_j.params
            resid_j = model_j.resid
            sigma2[j] = np.mean(resid_j**2)

        # EM loop
        ll_prev = -np.inf
        for it in range(self.max_iter):
            # E-step: responsibilities
            gamma = np.zeros((T, self.n_regimes))
            for t in range(T):
                xt = X[t]
                yt = Y[t]
                denom = 0.0
                for j in range(self.n_regimes):
                    mu_j = phi[j, 0] + np.dot(phi[j, 1:], xt)
                    pdf_j = norm.pdf(yt, mu_j, np.sqrt(sigma2[j] + 1e-10))  # avoid zero var
                    gamma[t, j] = pi[j] * pdf_j
                    denom += gamma[t, j]
                if denom > 0:
                    gamma[t] /= denom

            # M-step
            pi_new = np.mean(gamma, axis=0)
            for j in range(self.n_regimes):
                weights = gamma[:, j]
                sum_w = np.sum(weights)
                if sum_w < self.order + 1:
                    continue
                exog_full = add_constant(X)
                model_w = WLS(Y, exog_full, weights=weights).fit()
                phi[j] = model_w.params
                resid_w = model_w.resid
                sigma2[j] = np.sum(weights * resid_w**2) / sum_w

            pi = pi_new

            # Log-likelihood
            ll = 0.0
            for t in range(T):
                for j in range(self.n_regimes):
                    xt = X[t]
                    yt = Y[t]
                    mu_j = phi[j, 0] + np.dot(phi[j, 1:], xt)
                    pdf_j = norm.pdf(yt, mu_j, np.sqrt(sigma2[j] + 1e-10))
                    ll += gamma[t, j] * np.log(pi[j] * pdf_j + 1e-300)

            if abs(ll - ll_prev) < self.tol:
                print(f"Converged after {it+1} iterations")
                break
            ll_prev = ll

        self.pi = pi
        self.phi = phi
        self.sigma2 = sigma2
        self.gamma = gamma

        # Fitted values
        mu = np.zeros(T)
        for t in range(T):
            xt = X[t]
            for j in range(self.n_regimes):
                mu[t] += gamma[t, j] * (phi[j, 0] + np.dot(phi[j, 1:], xt))
        self.fittedvalues = np.full(n, np.nan)
        self.fittedvalues[self.order:] = mu
        self.resid = data - self.fittedvalues

        # AIC
        ssr = np.sum(self.resid[~np.isnan(self.resid)]**2)
        k = self.n_regimes * (self.order + 2) - 1  # phi (order+1), sigma2, pi (n_regimes-1)
        self.aic = T * np.log(ssr / T) + 2 * k