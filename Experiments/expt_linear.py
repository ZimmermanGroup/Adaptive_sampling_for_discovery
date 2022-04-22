import numpy as np

import importlib

from utils import *
from linear_utils import *
import expt_utils as expt_utils
# importlib.reload(expt_utils)


algos_dict = {
  'IDS': [IDS_linear, {"M": 100}, 10, 'IDS', 'm'],
  'TS': [TS_linear, {}, 10, 'TS', 'b'],
  'UCB': [UCB_linear, {}, 10, 'UCB', 'y'],
  # 'Random': [linear_solver, {}, 10, 'Random', '-sg']
}

prob_dict = {
  "m": 500,
  "noise": [0.1, 1.0, 5.0, 10.0],
  "d": 20
}

script_file = 'Experiments/expt_linear.py'

# expt_utils.algos_metrics(
#     prob_dict, algos_dict, linear_discover, T = 500,
#     results_dir='results/linear/', script_file=script_file
#     )

expt_utils.algos_vs_var_metrics(
    prob_dict, algos_dict, linear_discover, T = 100, log_y = True,
    results_dir='results/linear/', script_file=script_file, load = "/Users/zipingxu/Desktop/Research/Discovery/Code/Code/selected_results/Linear/Noise/IDS-TS-UCB_vs_20220406-153739/output_20220406-153739.pkl"
    )
