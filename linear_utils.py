"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Mar 21, 2022
Record  :    Utilise for linear model
"""

from utils import discover
from numpy.random import randn
from copy import deepcopy
import pdb
import numpy as np


class linear_discover(discover):
    def __init__(self, m, d, sigma = 1.0, sig_coef = 1.0, noise = 0.1):
        super(linear_discover, self).__init__()
        self.m = m
        self.d = d
        self.sigma = sigma
        self.sig_coef = sig_coef
        self.noise = noise
        self.init_linear()
    def init_linear(self):
        self.x = randn(self.m * self.d).reshape(self.m, self.d) * self.sigma
        self.y_noise = randn(self.m) * self.noise
        self.sig_coef = randn(self.d) * self.sig_coef
        self.y = self.x @ self.sig_coef.transpose() + self.y_noise
    def get_reward(self, action):
        return self.y[action], None
    def get_mask(self):
        mask_mat = np.array([False]*(self.m))
        for action in self.actions:
            mask_mat[action] = True
        self.mask_mat = mask_mat
        return mask_mat
    def regret(self):
        T = len(self.rewards)
        xf = -self.y.flatten()
        xf.sort()
        reg = -(np.cumsum(xf[:T])) - np.cumsum(self.rewards)
        return reg

class linear_solver:
    def __init__(self, prob):
        self.prob = prob
    def step(self):
        pass
    def run(self, T):
        self.prob.refresh()
        acts = np.random.choice(range(self.prob.m), T, replace=False)
        for t in range(T):
            self.prob.take_action(acts[t])
        return self.prob.regret(), self.prob.rewards

class TS_linear(linear_solver):
    def __init__(self, prob):
        super(TS_linear, self).__init__(prob)
        self.init_priors()
    def init_priors(self, s0 = 1.0):
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        f = self.prob.x[action, :]
        s_inv = np.linalg.inv(self.sigma_t)
        ffT = np.outer(f, f)
        # pdb.set_trace()
        mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.prob.noise**2), np.dot(s_inv, self.mu_t) + r * f / self.prob.noise**2)
        sigma_ = np.linalg.inv(s_inv + ffT/self.prob.noise**2)
        self.mu_t = mu_
        self.sigma_t = sigma_
        return mu_, sigma_
    def step(self):
        theta_t = np.random.multivariate_normal(self.mu_t, self.sigma_t, 1)
        pred = self.prob.x @ theta_t.transpose()
        self.prob.get_mask()
        pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
        return self.prob.regret(), self.prob.rewards


class UCB_linear(linear_solver):
    def __init__(self, prob, lbda = 10e-4, alpha = 10e-1):
        super(UCB_linear, self).__init__(prob)
        self.A_t = lbda * np.eye(self.prob.d)
        self.b_t = np.zeros(self.prob.d)
        self.lbda = lbda
        self.alpha = alpha
    def step(self):
        inv_A = np.linalg.inv(self.A_t)
        theta_t = np.dot(inv_A, self.b_t)
        beta_t = self.alpha * np.sqrt(np.diagonal(np.dot(np.dot(self.prob.x, inv_A), self.prob.x.T)))
        pred = np.dot(self.prob.x, theta_t) + beta_t
        self.prob.get_mask()
        pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at
    def update_est(self, a_t, r):
        self.A_t += np.outer(self.prob.x[a_t, :], self.prob.x[a_t, :])
        self.b_t += r * self.prob.x[a_t, :]
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_est(action, r)
        return self.prob.regret(), self.prob.rewards

class IDS_linear(linear_solver):
    def __init__(self, prob, M = 1000):
        super(IDS_linear, self).__init__(prob)
        self.M = M
        self.init_priors()
    def init_priors(self, s0 = 1.0):
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        f = self.prob.x[action, :]
        s_inv = np.linalg.inv(self.sigma_t)
        ffT = np.outer(f, f)
        # pdb.set_trace()
        mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.prob.noise**2), np.dot(s_inv, self.mu_t) + r * f / self.prob.noise**2)
        sigma_ = np.linalg.inv(s_inv + ffT/self.prob.noise**2)
        self.mu_t = mu_
        self.sigma_t = sigma_
        return mu_, sigma_
    def step(self):
        """
        Implementation of linearSampleVIR (algorithm 6 in Russo & Van Roy, p. 244) applied for Linear  Bandits with
        multivariate normal prior. Here integrals are approximated in sampling thetas according to their respective
        posterior distributions.
        :param mu_t: np.array, posterior mean vector at time t
        :param sigma_t: np.array, posterior covariance matrix at time t
        :param M: int, number of samples
        :return: int, np.array, arm chose and p*
        """
        mu_t, sigma_t, M = self.mu_t, self.sigma_t, self.M
        self.n_a = self.prob.m
        self.prob.get_mask()
        thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
        mu = np.mean(thetas, axis=0)
        # print(self.features.shape, thetas.shape)
        means = np.dot(self.prob.x, thetas.T)
        means[self.prob.mask_mat] = np.nan
        theta_hat = np.nanargmax(means, axis=0)
        theta_hat_ = [thetas[np.where(theta_hat==a)] for a in range(self.n_a)]
        p_a = np.array([len(theta_hat_[a]) for a in range(self.n_a)])/M
        mu_a = np.nan_to_num(np.array([np.nanmean([theta_hat_[a]], axis=1).squeeze() for a in range(self.n_a)]))
        L_hat = np.nansum(np.array([p_a[a]*np.outer(mu_a[a]-mu, mu_a[a]-mu) for a in range(self.n_a)]), axis=0)
        rho_star = np.nansum(np.array([p_a[a]*np.dot(self.prob.x[a], mu_a[a]) for a in range(self.n_a)]), axis=0)
        # v = np.array([np.dot(np.dot(self.prob.x[a], np.dot(sigma_t, sigma_t.T)), self.prob.x[a].T) for a in range(self.n_a)]) + 0.000001
        v = np.array([np.dot(np.dot(self.prob.x[a], L_hat), self.prob.x[a].T) for a in range(self.n_a)]) + 0.000001
        delta = np.array([rho_star - np.dot(self.prob.x[a], mu) for a in range(self.n_a)])
        delta[self.prob.mask_mat] = np.nan
        arm = np.nanargmax(-delta**2/v)
        # pdb.set_trace()
        return arm
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
        return self.prob.regret(), self.prob.rewards
