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
import bayes_logistic as bl
import numpy as np

expit = lambda x: 1/(1+np.exp(-x))

class logistic_discover(discover):
    def __init__(self, m, d, sigma = 1.0, sig_coef = 1.0, noise = 0.1):
        super(logistic_discover, self).__init__()
        self.m = m
        self.d = d
        self.sigma = sigma
        self.noise = noise
        self.sig_coef = sig_coef
        self.init_linear()
    def init_linear(self):
        self.x = randn(self.m * self.d).reshape(self.m, self.d) * self.sigma
        self.sig_coef = randn(self.d) * self.sig_coef
        self.y_expt = np.apply_along_axis(expit, 0, self.x @ self.sig_coef.transpose())
        self.y = np.array([np.random.choice([0, 1], size = 1, p = [1-p, p])[0] for p in self.y_expt])
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

class logistic_solver:
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

class TS_logistic(logistic_solver):
    def __init__(self, prob, s0 = 1.0):
        super(TS_logistic, self).__init__(prob)
        self.s0 = s0
        self.init_priors()
    def init_priors(self):
        self.mu_0 = np.zeros(self.prob.d)
        self.sigma_0 = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        mu_, sigma_ = bl.fit_bayes_logistic(np.array(self.prob.y[self.prob.actions]), self.prob.x[self.prob.actions, :], self.mu_0, self.sigma_0)
        # f = self.prob.x[action, :]
        # s_inv = np.linalg.inv(self.sigma_t)
        # ffT = np.outer(f, f)
        # # pdb.set_trace()
        # mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.prob.noise**2), np.dot(s_inv, self.mu_t) + r * f / self.prob.noise**2)
        # sigma_ = np.linalg.inv(s_inv + ffT/self.prob.noise**2)
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
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards

class UCB_logistic(logistic_solver):
    def __init__(self, prob, alpha = 1.0, delta = 1.0, sigma = 1.0):
        super(UCB_logistic, self).__init__(prob)
        self.dimension=prob.d
        # self.iteration=iteration
        self.item_num=self.prob.m
        self.item_feature=self.prob.x
        self.true_payoffs=self.prob.y
        self.alpha=alpha
        self.delta=delta
        self.sigma=sigma
        self.beta=1.0
        self.cov=self.alpha*np.identity(self.dimension)
        self.bias=np.zeros(self.dimension)
        self.user_f=np.zeros(self.dimension)
        # self.item_index=np.zeros(self.iteration)

    def random_select(self):

        index=np.random.choice(range(self.item_num))
        while index in self.prob.actions:
            index=np.random.choice(range(self.item_num))
        x=self.item_feature[index]
        noise=np.random.normal(scale=self.sigma)
        payoff=self.true_payoffs[index]+noise
        #self.prob.get_mask()
        #t_y = self.true_payoffs
        #t_y[self.prob.mask_mat] = np.nan
        regret=np.nanmax(self.true_payoffs)-self.true_payoffs[index]
        return x, payoff, regret, index

    def update_beta(self, time):
        self.beta=np.sqrt(self.alpha)+self.sigma*np.sqrt(self.dimension*np.log(1+time/self.dimension)+2*np.log(1/self.delta))

    def step(self):
        index_list=np.zeros(self.item_num)
        cov_inv=np.linalg.pinv(self.cov)
        for i in range(self.item_num):
            x=self.item_feature[i]
            x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
            est_y=np.dot(self.user_f, x)
            index_list[i]=est_y+self.beta*x_norm
        self.prob.get_mask()
        index_list[self.prob.mask_mat] = np.nan
        index=np.nanargmax(index_list)
        return index

    def update_feature(self, x,y):
        self.cov+=np.outer(x,x)
        self.bias+=x*y
        self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)

    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            if t<=self.dimension:
                x, y, regret, index=self.random_select()
                r, _ = self.prob.take_action(index)
                self.update_feature(x,y)
            else:
                self.update_beta(t)
                action=self.step()
                r, _ = self.prob.take_action(action)
                self.update_feature(self.prob.x[action],r)
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards


class IDS_logistic(logistic_solver):
    def __init__(self, prob, M = 1000, s0 = 1.0):
        super(IDS_logistic, self).__init__(prob)
        self.M = M
        self.init_priors()
    def init_priors(self, s0 = 1.0):
        self.mu_0 = np.zeros(self.prob.d)
        self.sigma_0 = s0 * np.eye(self.prob.d)
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        # pdb.set_trace()
        mu_, sigma_ = bl.fit_bayes_logistic(np.array(self.prob.y[self.prob.actions]), self.prob.x[self.prob.actions, :], self.mu_0, self.sigma_0)
        self.mu_t = mu_
        self.sigma_t = sigma_
        # print(mu_, sigma_)
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
        # np.apply_along_axis(expit, 0, self.x @ self.sig_coef.transpose())
        means = expit(np.dot(self.prob.x, thetas.T))
        means[self.prob.mask_mat] = np.nan
        theta_hat = np.nanargmax(means, axis=0)
        theta_hat_ = [thetas[np.where(theta_hat==a)] for a in range(self.n_a)]
        # pdb.set_trace()
        p_a = np.array([len(theta_hat_[a]) for a in range(self.n_a)])/M
        mu_a = np.nan_to_num(np.array([np.nanmean([theta_hat_[a]], axis=1).squeeze() for a in range(self.n_a)]))
        L_hat = np.nansum(np.array([p_a[a]*np.outer(mu_a[a]-mu, mu_a[a]-mu) for a in range(self.n_a)]), axis=0)

        rho_star = np.nansum(np.array([p_a[a]* expit(np.dot(self.prob.x[a], mu_a[a])) for a in range(self.n_a)]), axis=0)
        # v = np.array([np.dot(np.dot(self.prob.x[a], np.dot(sigma_t, sigma_t.T)), self.prob.x[a].T) for a in range(self.n_a)]) + 0.000001
        v = np.array([np.dot(np.dot(self.prob.x[a], L_hat), self.prob.x[a].T) for a in range(self.n_a)]) + 0.000001
        f_prime = lambda x: -np.exp(-x) / (1+np.exp(-x))**2
        # pdb.set_trace()
        first_order = f_prime(np.dot(self.prob.x, mu))**2
        v = np.multiply(v, first_order)
        # v = np.array([np.dot(np.dot(self.features[a], L_hat), self.features[a].T) for a in range(self.n_a)]) + 0.000001
        delta = np.array([rho_star - expit(np.dot(self.prob.x[a], mu)) for a in range(self.n_a)])
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
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards
