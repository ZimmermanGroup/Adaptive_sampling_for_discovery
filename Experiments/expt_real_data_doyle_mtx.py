import numpy as np

import importlib
import pandas as pd
from utils import *
from utils import matrix_discover
import expt_utils as expt_utils
# importlib.reload(expt_utils)


algos_dict = {
  'IDS': [IDS_matrix,
    {"gamma": 3, "sigma": 0.01, 'update': 5}, 1, 'IDS', '-^m'],
  'Greedy': [IDS_matrix,
    {"gamma": 3, "sigma": 0.01, 'update': 5, 'greedy': True}, 1, 'Greedy', '-^b'],
  'Random': [matrix_solver, {}, 1, 'Random', '^g']
}

xls = pd.ExcelFile('data/doyle_matrix.xls')
df_doyle = pd.read_excel(xls, 'doyle_matrix')
df_desc = pd.read_excel(xls, 'doyle_desc')
df_add_desc = pd.read_excel(xls, 'additive_desc')
df_aryl_desc = pd.read_excel(xls, 'aryl_desc')

n_row, n_col = df_doyle.shape
_, n_desc =  df_desc.shape


for i in range(n_col):
  prob_dict = {
    "m1": 22,
    "m2": 15,
    "R": 4,
    "noise": 0.1
  }
  prob = matrix_discover(**prob_dict)
  prob.x = (df_doyle.iloc[:, i].values / 100.0).reshape((22, 15))
  prob.x_noise = (df_doyle.iloc[:, i].values / 100.0).reshape((22, 15))

  script_file = 'expt_real_data_doyle_mtx.py'

  expt_utils.algos_real_data(
    prob, algos_dict, matrix_discover, T = 330,
    results_dir='results/doyle_mtx', script_file=script_file
    )



# x_dat = x_dat / 100 - 0.5
# prob.x = x_dat
# prob.x_noise = x_dat
# prob.noise = 0

# script_file = 'expt_real_data.py'

# expt_utils.algos_real_data(
#     prob, algos_dict, matrix_discover, T = 80,
#     results_dir='results/matrix', script_file=script_file
#     )
