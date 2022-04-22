import numpy as np

import importlib

from utils import *
import expt_utils as expt_utils
# importlib.reload(expt_utils)


algos_dict = {
	'IDS': [IDS_matrix,
		{"gamma": 3, "sigma": 0.1, 'update': 1}, 10, 'IDS', 'm'],
	'Greedy': [IDS_matrix,
		{"gamma": 3, "sigma": 0.1, 'update': 1, 'greedy': True}, 10, 'TS', 'b'],
	'Random': [matrix_solver, {}, 10, 'Random', 'g']
}

prob_dict = {
	"m1": 30,
	"m2": 30,
	"R": 2,
	"noise": [0.1, 0.5, 1.0]
}

script_file = 'Experiments/expt_matrix.py'

expt_utils.algos_vs_var_metrics(
		prob_dict, algos_dict, matrix_discover, T = 100,#  xlims = (0, 150), ylims = (0, 700),
		results_dir='results/matrix', script_file=script_file, load = "/Users/zipingxu/Desktop/Research/Discovery/Code/Code/results/matrix/IDS-Greedy-Random_vs_20220407-175342/output_20220407-175342.pkl"
		)
