import numpy as np

import importlib

from utils import *
from graph_utils import *
import expt_utils as expt_utils
# importlib.reload(expt_utils)


algos_dict = {
  # 'Cheat': [Greedy_graph, {"cheat": True}, 100, 'Cheat', '^y'],
  'TS': [IDS_graph, {"n": 1, "TS": 1, 'update': 1}, 10, 'TS', 'b'],
  'IDS': [IDS_graph, {"n": 100, 'ga': 3, 'update': 1}, 10, 'IDS', 'm'],
  'Random': [graph_solver, {}, 10, 'random', 'g'],
}

prob_dict = {
  "m": 300,
  "sigma": 10.0,
  "noise": [0.1, 1.0, 10.0],
  "p": 0.01,
  "init": "Star"
}

script_file = 'Experiments/expt_graph.py'

expt_utils.algos_vs_var_metrics(
    prob_dict, algos_dict, graph_discover, T = 50, #log_y = True,
    results_dir='results/graph/', script_file=script_file, load = "/Users/zipingxu/Desktop/Research/Discovery/Code/Code/results/graph/TS-IDS-Random_vs_20220420-200626/output_20220420-200626.pkl"
    )
