import numpy as np

import importlib

from utils import *
from logistic_utils import *
from linear_utils import IDS_linear
import expt_utils as expt_utils
# importlib.reload(expt_utils)


algos_dict = {
  # 'IDS': [IDS_logistic, {"M": 1000}, 1, 'IDS', '-sm'],
  'TS': [TS_logistic, {}, 10, 'TS', 'b'],
  'Random': [logistic_solver, {}, 10, 'Random', 'g'],
  'UCB': [UCB_logistic, {}, 10, 'UCB', 'y'],
  'IDS': [IDS_logistic, {"M": 1000}, 10, 'IDS', 'm'],
  # 'IDS_Linear': [IDS_linear, {"M": 1000}, 10, 'IDS_Linear', '-sc'],
  # 'Random': [graph_solver, {}, 10, 'Random', '-sm']
}

prob_dict = {
  "m": 500,
  "d": [20, 50, 100]
}

script_file = 'Experiments/expt_logistic.py'

# expt_utils.algos_metrics(
#     prob_dict, algos_dict, linear_discover, T = 500,
#     results_dir='results/linear/', script_file=script_file
#     )

expt_utils.algos_vs_var_metrics(
    prob_dict, algos_dict, logistic_discover, T = 250,
    results_dir='results/logistic/', script_file=script_file, load = "/Users/zipingxu/Desktop/Research/Discovery/Code/Code/results/logistic/TS-Random-UCB-IDS_vs_20220407-225323/output_20220407-225323.pkl"
    )
