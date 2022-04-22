"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Mar 12, 2022
Record  :    Utilise for discovery
"""
import pdb
import numpy as np
from matrix_completion import svt_solve, pmf_solve, calc_unobserved_rmse
from numpy.random import randn
from copy import deepcopy

class discover:
    def __init__(self):
        self.initi_prob()
        self.actions = [] # labeled set
        self.rewards = [] # previous rewards
    def initi_prob(self):
        pass
    def refresh(self):
        self.actions = [] # labeled set
        self.rewards = [] # previous rewards
    def get_reward(self, action):
        pass
    def take_action(self, action):
        self.actions.append(action)
        reward, side_info = self.get_reward((action))
        self.rewards.append(reward)
        return self.rewards[-1], side_info
    def regret(self):
        pass




class matrix_discover(discover):
    def __init__(self, m1, m2, R, noise = 0.1, pre_mask = None, decen = False):
        super(matrix_discover, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.R = R
        self.decen = decen
        self.noise = noise
        self.pre_mask = pre_mask
        self.init_prob()
    def init_prob(self):
        if not self.decen:
            self.x = randn(self.m1*self.R).reshape((self.m1, self.R)) @ randn(self.R*self.m2).reshape((self.R, self.m2))
            self.x_noise = self.x + randn(self.m1*self.m2).reshape(self.m1, self.m2) * self.noise
        else:
            self.U = randn(self.m1*self.R).reshape((self.m1, self.R))
            self.V = randn(self.R*self.m2).reshape((self.R, self.m2))
            mean_U = np.outer(np.ones(self.m1), np.mean(self.U, axis = 0))
            self.U = self.U - mean_U
            self.U = self.U / 10
            self.U += mean_U

            mean_V = np.outer(np.mean(self.V, axis = 1), np.ones(self.m2))
            self.V = self.V - mean_V
            self.V = self.V / 10
            self.V += mean_V

            self.x = self.U @ self.V
            self.x_noise = self.x + randn(self.m1*self.m2).reshape(self.m1, self.m2) * self.noise
    def get_indx(self, indx):
        return int(indx / self.m2), indx % self.m2
    def get_reward(self, action):
        a1, a2 = self.get_indx(action)
        side_info = None
        return self.x_noise[a1, a2], side_info
    def regret(self):
        T = len(self.rewards)
        # x_noise_mask = self.x_noise[]
        xf = -self.x_noise.flatten()
        xf.sort()
        reg = -(np.cumsum(xf[:T])) - np.cumsum(self.rewards)
        return reg
    def get_mask_xna(self):
        mask_mat = np.array([False]*(self.m1*self.m2)).reshape(self.m1, self.m2)
        for action in self.actions:
            mask_mat[int(action / self.m2), action % self.m2] = True
        xna = deepcopy(self.x_noise)
        xna[~mask_mat] = np.nan
        self.xna = xna
        self.mask_mat = mask_mat
        return mask_mat, xna
    def from_dat(self, dat_x):
        if not dat_x.shape == (self.m1, self.m2):
            assert "Incorrect dimensions"
        self.x = dat_x
        self.x_noise = dat_x

# random solver
class matrix_solver:
    def __init__(self, prob):
        self.prob = prob
        self.m1 = prob.m1
        self.m2 = prob.m2
        self.R = prob.R
    def step(self):
        pass
    def run(self, T):
        self.prob.refresh()
        if self.prob.pre_mask is None:
            acts = np.random.choice(range(self.m1*self.m2), T, replace=False)
            for t in range(T):
                self.prob.take_action(acts[t])
        else:
            ind_mat = np.array(range(self.m1 * self.m2)).reshape(self.m1, self.m2)
            ind_mat = ind_mat[~self.prob.pre_mask]
            # row_mat = np.outer((np.array(range(self.m1)) + 1),  (np.zeros(self.m2) + 1))
            # col_mat = np.outer((np.zeros(self.m1) + 1), (np.array(range(self.m2)) + 1))
            # row_ind = row_mat[~self.pre_mask]
            # col_ind = row_mat[~self.pre_mask]
            nsample = len(ind_mat)
            print(nsample)
            acts = np.random.choice(range(nsample), T, replace=False)
            for t in range(T):
                self.prob.take_action(ind_mat[acts[t]])
        return self.prob.regret(), self.prob.rewards

class IDS_matrix(matrix_solver):
    def __init__(self, prob, sigma = 1, Nmcmc = 500, burnin = 100, update = 5, gamma = 2, greedy = False):
        super(IDS_matrix, self).__init__(prob)
        self.sigma = sigma
        self.lam = 1/(4*sigma**2)
        self.Nmcmc = Nmcmc
        self.burnin = burnin
        self.update = update
        self.ga = gamma
        self.greedy = greedy
        self.ratio_flag = False
    def step(self):
        # just for bug free
        if len(self.prob.actions) < 5:
            a = np.random.choice(range(self.m1*self.m2), 1, replace=False)[0]
            while a in self.prob.actions:
                a = np.random.choice(range(self.m1*self.m2), 1, replace=False)[0]
            return a
        mask_mat, xna = self.prob.get_mask_xna()
        if len(self.prob.actions) % self.update > 0 and self.ratio_flag > 0:
            self.ratio[mask_mat] = np.nan
            # pdb.set_trace()
            a = np.nanargmin(self.ratio)
            return a
        I1 = ((np.zeros_like(xna.transpose(), dtype=int) + 1) * np.array(range(self.m1))).transpose()
        I2 = (np.zeros_like(xna, dtype=int) + 1) * np.array(range(self.m2))
        I1 = I1[mask_mat]
        I2 = I2[mask_mat]

        Y = xna[mask_mat]
        nsample = len(self.prob.actions)
        Mstep = np.zeros((self.m1, self.R)) + 1.0
        Nstep = np.zeros((self.m2, self.R)) + 1.0
        gamma = [1] * self.R
        datalength = np.array(range(nsample))

        # MCMC
        res = []
        for step in range(self.Nmcmc+self.burnin):
            # update M[i,j]
            for i in range(self.m1):
                # pdb.set_trace()
                try:
                    seti = datalength[I1==i]
                except IndexError:
                    pdb.set_trace()
                # seti = seti[~np.isnan(seti)]
                for j in range(self.R):
                    Msteptrouj = Mstep[i, :]
                    Msteptrouj[j] = 0
                    # pdb.set_trace()
                    V = (1/gamma[j]) + self.lam*np.sum(Nstep[I2[seti],j]**2)
                    D = sum(self.lam*(Y[seti] - Msteptrouj @ (Nstep[I2[seti], :].transpose()) ) *Nstep[I2[seti],j])
                    Mstep[i,j] = (randn(1) + D/np.sqrt(V)) / np.sqrt(V)
            # update N[i,j]
            for i in range(self.m2):
                seti = datalength[I2==i]
                # seti = seti[~np.isnan(seti)]
                for j in range(self.R):
                    Nsteptrouj = Nstep[i, :]
                    Nsteptrouj[j] = 0
                    V = (1/gamma[j]) + self.lam*sum(Mstep[I1[seti], j]**2)
                    D = sum(self.lam*(Y[seti] - Nsteptrouj @ (Mstep[I1[seti], :]).transpose() )*Mstep[I1[seti],j])
                    Nstep[i,j] = (randn(1) + D/np.sqrt(V)) / np.sqrt(V)
            if step >= self.burnin:
                tmp = Mstep @ Nstep.transpose()
                tmp[mask_mat] = np.nan
                if self.prob.pre_mask is not None:
                    tmp[self.prob.pre_mask] = np.nan
                res.append(tmp)

        # take action
        indx = [np.nanargmax(M) for M in res]
        ind_unq, ind_cnt = np.unique(indx, return_counts=True)
        mean_mtx = []
        res = np.array(res)
        for ind in ind_unq:
            mean_mtx.append(np.mean(res[indx == ind, :, :], axis = 0))
        mean_mtx = np.array(mean_mtx)
        mean_total = np.mean(res, axis = 0)
        problty = np.expand_dims(np.expand_dims(ind_cnt / np.sum(ind_cnt), 1), 2)
        cond_var = np.sum((mean_mtx - mean_total)**2 * problty, axis = 0) + 0.000001
        delta = np.array([mean_mtx[i, int(ind_unq[i] / self.m2), ind_unq[i] % self.m2] - mean_mtx[i, :, :] for i in range(len(ind_unq))])
        delta = np.sum(delta * problty, axis = 0)

        self.ratio = delta**self.ga / cond_var
        self.ratio_flag = True
        if ~self.greedy:
            a = np.nanargmin(self.ratio)
        else:
            a = np.nanargmin(delta)
        return a
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            action = self.step()
            self.prob.take_action(action)
        return self.prob.regret(), self.prob.rewards




class EC_matrix(matrix_solver):
    def __init__(self, prob, thrd = 0.7, update = 5):
        super(EC_matrix, self).__init__(prob)
        self.thrd = thrd
        self.update = update
        self.x_hat = None
    def step(self):
        mask_mat, xna = self.prob.get_mask_xna()
        if len(self.prob.actions) % self.update > 0 and self.x_hat is not None:
            self.x_hat[mask_mat] = -100
            a = np.argmax(self.x_hat)
            return a
        try:
            self.x_hat = svt_solve(self.prob.x_noise, (mask_mat.astype(int)))
            self.x_hat[mask_mat] = -100
            a = np.argmax(self.x_hat)
        except ValueError:
            a = np.random.choice(range(self.m1*self.m2), 1, replace=False)[0]
            return a
        return a
    def run(self, T):
        self.prob.refresh()
        acts = np.random.choice(range(self.m1*self.m2), int(self.thrd * T), replace=False)
        for t in range(int(self.thrd * T)):
            self.prob.take_action(acts[t])
        for t in range(T - int(self.thrd * T)):
            action = self.step()
            self.prob.take_action(action)
        return self.prob.regret(), self.prob.rewards


# prob = discover()
# solv = solver(prob)
# for t in range(T):
#     action = solv.step()
#     prob.take_action(action)
# prob.regret()










