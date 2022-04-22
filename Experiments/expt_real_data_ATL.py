import numpy as np
import joblib
import importlib

import pandas as pd
from utils import *
from logistic_utils import *
from linear_utils import *
import expt_utils as expt_utils
from sklearn.preprocessing import normalize
# importlib.reload(expt_utils)

N_trial = 10
algos_dict = {
  'Random': [logistic_solver, {}, N_trial, 'Random', 'g'],
  'IDS': [IDS_logistic, {"M": 1000, "s0": 0.1}, N_trial, 'IDS', 'm'],
  'TS': [TS_logistic, {"s0": 0.1}, N_trial, 'TS', 'b'],
  'UCB': [UCB_logistic, {'sigma': 0.1}, N_trial, 'UCB', 'y'],
  # 'Random': [graph_solver, {}, 10, 'Random', '-sm']
}

x_dat = joblib.load("data/target_desc.joblib", mmap_mode=None)
y_dat = joblib.load("data/target_y_dict.joblib", mmap_mode=None)

cutoff = 8
for mol in x_dat:
  n_row, n_desc = x_dat[mol].shape
  prob_dict = {
    "m": n_row,
    "noise": 0.1,
    "d": cutoff
  }
  prob = logistic_discover(**prob_dict)
  x = x_dat[mol]# .to_numpy()
  x = x[:, :cutoff]
  x = normalize(x, axis=0, norm='l2')
  prob.x = x
  prob.y = -y_dat[mol]# df_doyle.iloc[:, i].values / 100.0

  script_file = 'Experiments/expt_real_data_ATL.py'

  expt_utils.algos_real_data(
      prob, algos_dict, linear_discover, T = n_row,
      results_dir='results/ATL/flipped', script_file=script_file
      )
