import os
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
    'Random': [linear_solver, {}, 50, 'Random', 'g'],
    'IDS': [IDS_linear, {'M': 1000}, 50, 'IDS', 'm'],
    'TS': [TS_linear, {}, 50, 'TS', 'b'],
    'UCB': [UCB_linear, {}, 50, 'UCB', 'y'],
    # 'Random': [graph_solver, {}, 10, 'Random', '-sm']
}

prob_dict = {'m': 80, 'noise': 1.0, 'd': 5}

x_dat = pd.read_csv('data/11_des.csv', header=None).to_numpy()
y_dat = pd.read_csv('data/11_yield.csv', header=None).to_numpy()

for i in range(11):
    prob = linear_discover(**prob_dict)
    x = x_dat
    x = normalize(x, axis=0, norm='l2')
    prob.x = x
    prob.y = y_dat[:, i] / 100.0

    # script_file = 'expt_real_data_11columns.py'
    script_file = os.path.realpath(__file__)

    # expt_utils.algos_metrics(
    #     prob_dict, algos_dict, linear_discover, T = 500,
    #     results_dir='results/linear/', script_file=script_file
    #     )

    expt_utils.algos_real_data(
        prob,
        algos_dict,
        linear_discover,
        T=80,
        results_dir='results/11_columns/',
        script_file=script_file,
    )
