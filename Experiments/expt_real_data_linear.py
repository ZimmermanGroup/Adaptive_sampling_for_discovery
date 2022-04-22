import numpy as np
import joblib
import importlib

from utils import *
from logistic_utils import *
from linear_utils import IDS_linear
import expt_utils as expt_utils
# importlib.reload(expt_utils)


algos_dict = {
  'Random': [logistic_solver, {}, 10, 'Random', 'g'],
  'IDS': [IDS_logistic, {"M": 1000}, 10, 'IDS', 'm'],
  'TS': [TS_logistic, {}, 10, 'TS', 'b'],
  'UCB': [UCB_logistic, {}, 10, 'UCB', 'y'],
  # 'Random': [graph_solver, {}, 10, 'Random', '-sm']
}

prob_dict = {
  "m": 88,
  "noise": 0.0,
  "d": 8
}

y = np.array([1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 0., 1., 1., 1., 1.,
       1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0.,
       0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
       0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
       1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 0.,
       1., 1., 1.])

x = joblib.load("data/X_desc_full.joblib", mmap_mode=None)
x = x[:, np.array([0, 1, 4, 5, 6, 7, 10, 11])]
prob = logistic_discover(**prob_dict)
prob.x = x
prob.y = y

script_file = 'Experiments/expt_real_data_linear.py'

# expt_utils.algos_metrics(
#     prob_dict, algos_dict, linear_discover, T = 500,
#     results_dir='results/linear/', script_file=script_file
#     )

expt_utils.algos_real_data(
    prob, algos_dict, logistic_discover, T = 88,
    results_dir='results/logistic/', script_file=script_file
    )
