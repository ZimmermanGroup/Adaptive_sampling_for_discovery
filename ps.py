import numpy as np
from numpy.random import randn
from copy import deepcopy


m1 = 30
m2 = 30

J = 4
sig = 1
R = J
noise = 0.1

def get_indx(indx, m1 = m1, m2 = m2):
    return int(indx / m2), indx % m2


x = randn(m1*J).reshape((m1, J)) @ randn(J*m2).reshape((J, m2)) #+ 0.1*matrix(rnorm(n*50),nr=n)%*%matrix(rnorm(50*p),nc=p)
x_noise = x + randn(m1*m2).reshape(m1, m2) * noises
T = (J * m1) * 2
selected_set = [0]
action_set = list(range(m1*m2))
regrets = []
for t in range(T):
    print(t)
    # imiss = np.random.choice(range(m1*m2), int(m1*m2*missfrac), replace=False)
    imiss = list(set(action_set) - set(selected_set))
    mask_mat = np.array([True]*(m1*m2)).reshape(m1, m2)
    for ind in imiss:
        mask_mat[int(ind / m2), ind % m2] = False
    xna = deepcopy(x_noise)
    xna[~mask_mat] = np.nan


    #Gibbs for fix k
    I1 = ((np.zeros_like(xna.transpose(), dtype=int) + 1) * np.array(range(m1))).transpose()
    I2 = (np.zeros_like(xna, dtype=int) + 1) * np.array(range(m2))
    I1 = I1[mask_mat]
    I2 = I2[mask_mat]
    # I1 = row(xna)[!is.na(xna)]
    # I2 = col(xna)[!is.na(xna)]
    Y = xna[mask_mat]
    nsample =  m1*m2 - len(imiss)
    obs = Y
    sigma = 1

    k = J#min(n,p)/2 #rank of U,V
    lam = 1/(4*sigma^2)
    Mstep = np.zeros((m1, R)) + 1.0
    Nstep = np.zeros((m2, R)) + 1.0
    gamma = [1] * k
    Xmean = np.zeros((m1, m2))
    L2 = lam
    Nmcmc = 500
    burnin = 100
    datalength = np.array(range(nsample)) # as.vector(1:nsample)
    res = []
    for step in range(Nmcmc+burnin):
        # update M[i,j]
        for i in range(m1):
            seti = datalength[I1==i]
            seti = seti[~np.isnan(seti)]
            for j in range(k):
                Msteptrouj = Mstep[i, :]
                Msteptrouj[j] = 0
                V = (1/gamma[j]) + L2*np.sum(Nstep[I2[seti],j]**2)
                D = sum(L2*(obs[seti] - Msteptrouj @ (Nstep[I2[seti], :].transpose()) ) *Nstep[I2[seti],j])
                Mstep[i,j] = (randn(1) + D/np.sqrt(V)) / np.sqrt(V)
          # update N[i,j]
        for i in range(m2):
            seti = datalength[I2==i]
            seti = seti[~np.isnan(seti)]
            for j in range(k):
                Nsteptrouj = Nstep[i, :]
                Nsteptrouj[j] = 0
                V = (1/gamma[j]) + L2*sum(Mstep[I1[seti], j]**2)
                D = sum(L2*(obs[seti] - Nsteptrouj @ (Mstep[I1[seti], :]).transpose() )*Mstep[I1[seti],j])
                Nstep[i,j] = (randn(1) + D/np.sqrt(V)) / np.sqrt(V)
        if step >= burnin:
            tmp = Mstep @ Nstep.transpose()
            tmp[mask_mat] = -100
            res.append(tmp)
    indx = [np.argmax(M) for M in res]
    ind_unq, ind_cnt = np.unique(indx, return_counts=True)
    mean_mtx = []
    res = np.array(res)
    for ind in ind_unq:
        mean_mtx.append(np.mean(res[indx == ind, :, :], axis = 0))
    mean_mtx = np.array(mean_mtx)
    mean_total = np.mean(res, axis = 0)
    prob = np.expand_dims(np.expand_dims(ind_cnt / np.sum(ind_cnt), 1), 2)
    cond_var = np.sum((mean_mtx - mean_total)**2 * prob, axis = 0)
    delta = np.array([mean_mtx[i, int(ind_unq[i] / m2), ind_unq[i] % m2] - mean_mtx[i, :, :] for i in range(len(ind_unq))])
    delta = np.sum(delta * prob, axis = 0)

    a = np.argmin(delta**2 / cond_var)
    a1, a2 = get_indx(a, m1, m2)
    regrets.append(x_noise[a1, a2])
    selected_set.append(a)

from matrix_completion import svt_solve, calc_unobserved_rmse
T = (J * m1) * 2
selected_set = [0]
action_set = list(range(m1*m2))
regrets = []
thrd = 0.7
x_hat = None
for t in range(T):
    if t < int(T * thrd):
        a = np.random.choice(range(m1*m2), 1, replace=False)[0]
        a1, a2 = get_indx(a, m1, m2)
        regrets.append(x_noise[a1, a2])
        selected_set.append(a)
    else:
        imiss = list(set(action_set) - set(selected_set))
        mask_mat = np.array([True]*(m1*m2)).reshape(m1, m2)
        for ind in imiss:
            mask_mat[int(ind / m2), ind % m2] = False
        x_hat = svt_solve(x_noise, (mask_mat.astype(int)))
        x_hat[mask_mat] = -100
        a = np.argmax(x_hat)
        a1, a2 = get_indx(a, m1, m2)
        regrets.append(x_noise[a1, a2])
        selected_set.append(a)


xf = -x.flatten()
xf.sort()
reg = -(np.sum(xf[:T])) - np.sum(regrets)
