"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Mar 17, 2022
Record  :    Graph utils
"""
from utils import discover
from numpy.random import randn
from copy import deepcopy
import pdb
import numpy as np
import networkx.generators.random_graphs as rg
import networkx as nx

# possible initial graphs:
# star_graph

class graph_discover(discover):
    def __init__(self, m, sigma = 1.0, p = 0.5, subset = 5, noise = 0.1, init = 'Erdos_renyi'):
        super(graph_discover, self).__init__()
        self.m = m
        self.p = p
        self.sigma = sigma
        self.noise = noise
        self.subset = subset # for Complete_multipartite_graph
        self.init_graph(init)
    def init_graph(self, init):
        if init == "Erdos_renyi":
            self.graph = rg.erdos_renyi_graph(self.m, self.p)
            degree = np.array([de[1] for de in self.graph.degree()])
            self.x = (randn(self.m) * self.sigma + 1) #* degree # let high degree nodes to be worse
            self.x_noise = self.x + randn(self.m) * self.noise
        elif init == "Complete":
            self.graph = nx.complete_graph(self.m)
            degree = np.array([de[1] for de in self.graph.degree()])
            self.x = (randn(self.m) * self.sigma + 1) #* degree # let high degree nodes to be worse
            self.x_noise = self.x + randn(self.m) * self.noise
        elif init == "Complete_multipartite_graph":
            subm = int(self.m / self.subset)
            sizes = [subm] * self.subset + [self.m%self.subset]
            self.graph = nx.complete_multipartite_graph(sizes)
            degree = np.array([de[1] for de in self.graph.degree()])
            self.x = (randn(self.m) * self.sigma + 1) #* degree # let high degree nodes to be worse
            self.x_noise = self.x + randn(self.m) * self.noise
        elif init == "Power":
            self.graph = nx.powerlaw_cluster_graph(self.m, m = int(self.m / 10), p = self.p)
            self.x = (randn(self.m) * self.sigma ) #* degree # let high degree nodes to be worse
            self.x_noise = self.x + randn(self.m) * self.noise
        elif init == "Star":
            rm = int(self.m * (2.0 / 3))
            self.graph = rg.erdos_renyi_graph(rm, self.p)
            for i in range(self.m - rm):
                self.graph.add_node(i + rm)
            for i in range(self.m - rm):
                for j in range(self.m):
                    self.graph.add_edge(i + rm, j)
            degree = np.array([de[1] for de in self.graph.degree()])
            self.x = (randn(self.m) * self.sigma + 0.0) #/ degree # let high degree nodes to be worse
            # self.x[int(self.m * 2 / 3):self.m] -= 1.0
            self.x_noise = self.x + randn(self.m) * self.noise

    def get_reward(self, action):
        rewards = [(action, self.x_noise[action])]
        for edge in self.graph.edges([action]):
            rewards.append((edge[1], self.x[edge[1]] + randn(1) * self.noise))
        return self.x_noise[action], rewards
    def get_mask(self):
        mask_mat = np.array([False]*(self.m))
        for action in self.actions:
            mask_mat[action] = True
        self.mask_mat = mask_mat
        return mask_mat
    def regret(self):
        T = len(self.rewards)
        xf = -self.x_noise.flatten()
        xf.sort()
        reg = -(np.cumsum(xf[:T])) - np.cumsum(self.rewards)
        return reg

class graph_solver:
    def __init__(self, prob):
        self.prob = prob
    def step(self):
        pass
    def run(self, T):
        self.prob.refresh()
        acts = np.random.choice(range(self.prob.m), T, replace=False)
        # print(acts)
        for t in range(T):
            self.prob.take_action(acts[t])
        return self.prob.regret(), self.prob.rewards

class Greedy_graph(graph_solver):
    def __init__(self, prob, cheat = False, update = 5):
        super(Greedy_graph, self).__init__(prob)
        self.act_buffer = {act:[0, 0] for act in range(self.prob.m)}
        self.cheat = cheat
        self.update = update
        self.mu1 = None
    def current_mean(self):
        ps = []
        self.sig1 = []
        self.mu1 = []
        for act in range(self.prob.m):
            sig1 = 1 / (1/self.prob.sigma + self.act_buffer[act][1] / self.prob.noise)
            mu1 = sig1 * (self.act_buffer[act][0] / self.prob.noise)
            self.sig1.append(sig1)
            self.mu1.append(mu1)
        # if len(self.prob.actions) == 52:
            # print(self.cheat)
            # print(np.sum(self.sig1))
        return np.array(self.mu1)
    def step(self):
        if len(self.prob.actions) % self.update > 0 and self.mu1 is not None:
            self.prob.get_mask()
            self.mu1[self.prob.mask_mat] = np.nan
            a = np.nanargmin(self.mu1)
            return a
        self.prob.get_mask()
        mu1 = self.current_mean()
        mu1[self.prob.mask_mat] = np.nan
        self.mu1 = mu1
        a = np.nanargmax(mu1)
        return a
    def update_buffer(self, rewards):
        for act, rwd in rewards:
            self.act_buffer[act][0] += rwd
            self.act_buffer[act][1] += 1
    def run(self, T):
        self.prob.refresh()
        if not self.cheat:
            for t in range(T):
                action = self.step()
                _, tmp_rewards = self.prob.take_action(action)
                self.update_buffer(tmp_rewards)
            return self.prob.regret(), self.prob.rewards
        else:
            # print("I am cheating")
            acts = np.random.choice(range(2*int(self.prob.m/3), self.prob.m), 1, replace=False)
            for t in range(1):
                _, tmp_rewards = self.prob.take_action(acts[t])
                self.update_buffer(tmp_rewards)
            for t in range(T - 1):
                action = self.step()
                _, tmp_rewards = self.prob.take_action(action)
                self.update_buffer(tmp_rewards)
            return self.prob.regret(), self.prob.rewards


class IDS_graph(graph_solver):
    def __init__(self, prob, n = 100, ga = 3, update = 5, TS = False):
        super(IDS_graph, self).__init__(prob)
        self.ga = ga
        self.n = n
        self.TS = TS
        self.update = update
        self.ratio = None
        self.act_buffer = {act:[0, 0] for act in range(self.prob.m)}
    def posterior_sample(self, n):
        ps = []
        self.sig1 = []
        self.mu1 = []
        for act in range(self.prob.m):
            sig1 = 1 / (1/self.prob.sigma + self.act_buffer[act][1] / self.prob.noise)
            mu1 = sig1 * (self.act_buffer[act][0] / self.prob.noise)
            ps.append(randn(n) * sig1 + mu1)
            self.sig1.append(sig1)
            self.mu1.append(mu1)
        ps = np.array(ps).transpose()
        # pdb.set_trace()
        ps[:, self.prob.mask_mat] = np.nan
        return ps
    def step(self):
        if len(self.prob.actions) % self.update > 0 and self.ratio is not None:
            self.prob.get_mask()
            self.ratio[self.prob.mask_mat] = np.nan
            a = np.nanargmin(self.ratio)
            return a
        self.prob.get_mask()
        ps = self.posterior_sample(self.n)
        indx = np.nanargmax(ps, axis = 1)
        if not self.TS:
            ind_unq, ind_cnt = np.unique(indx, return_counts=True)
            mean_mtx = []
            for ind in ind_unq:
                mean_mtx.append(np.mean(ps[indx == ind, :], axis = 0))
            mean_mtx = np.array(mean_mtx)
            mean_total = np.mean(ps, axis = 0)
            problty = np.expand_dims(ind_cnt / np.sum(ind_cnt), 1)
            cond_var = np.sum((mean_mtx - mean_total)**2 * problty, axis = 0)
            act_var = []
            for act in range(self.prob.m):
                act_var.append(
                    np.nansum([cond_var[act2] for _, act2 in self.prob.graph.edges([act])] + [cond_var[act]]))
            act_var = np.array(act_var)
            delta = np.array([mean_mtx[i, ind_unq[i]] - mean_mtx[i, :] for i in range(len(ind_unq))])
            delta = np.sum(delta * problty, axis = 0)
            if np.sum(act_var) == 0:
                return np.nanargmin(delta)
            self.ratio = delta**self.ga / act_var
            a = np.nanargmin(self.ratio)
        else:
            a = indx[0]
        return a
    def update_buffer(self, rewards):
        for act, rwd in rewards:
            self.act_buffer[act][0] += rwd
            self.act_buffer[act][1] += 1
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            action = self.step()
            _, tmp_rewards = self.prob.take_action(action)
            self.update_buffer(tmp_rewards)
        return self.prob.regret(), self.prob.rewards


