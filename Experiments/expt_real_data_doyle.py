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


algos_dict = {
  'Random': [linear_solver, {}, 10, 'Random', 'g'],
  'IDS': [IDS_linear, {"M": 1000}, 10, 'IDS', 'm'],
  'TS': [TS_linear, {}, 10, 'TS', 'b'],
  'UCB': [UCB_linear, {}, 10, 'UCB', 'y'],
  # 'Random': [graph_solver, {}, 10, 'Random', '-sm']
}


xls = pd.ExcelFile('data/doyle_matrix.xls')
df_doyle = pd.read_excel(xls, 'doyle_matrix')
df_desc = pd.read_excel(xls, 'doyle_desc')
df_add_desc = pd.read_excel(xls, 'additive_desc')
df_aryl_desc = pd.read_excel(xls, 'aryl_desc')

n_row, n_col = df_doyle.shape
_, n_desc =  df_desc.shape
cutoff = n_desc
for i in range(n_col):
  prob_dict = {
    "m": n_row,
    "noise": 0.01,
    "d": cutoff
  }
  prob = linear_discover(**prob_dict)
  x = df_desc.to_numpy()
  x = x[:, :cutoff]
  x = normalize(x, axis=0, norm='l2')
  prob.x = x
  prob.y = df_doyle.iloc[:, i].values / 100.0

  script_file = 'Experiments/expt_real_data_doyle.py'

  expt_utils.algos_real_data(
      prob, algos_dict, linear_discover, T = n_row,
      results_dir='results/doyle/', script_file=script_file
      )
