import numpy as np

import importlib
import pandas as pd
from utils import *
from utils import matrix_discover
import expt_utils as expt_utils
# importlib.reload(expt_utils)


algos_dict = {
  'Random': [matrix_solver, {}, 5, 'Random', 'sg'],
  'IDS': [IDS_matrix,
    {"gamma": 3, "sigma": 1.0, 'update': 5}, 5, 'IDS', 'sm'],
  'TS': [IDS_matrix,
    {"gamma": 3, "sigma": 1.0, 'update': 5, 'greedy': True}, 5, 'TS', 'sb']
}
dat = pd.read_csv("data/ml-100k/u.data", sep = "\t", header = None)
x = np.zeros((943, 1682))
for i in range(100000):
  user_id, item_id, rate, _ = dat.iloc[i, :]
  x[user_id-1, item_id-1] = rate

# import matplotlib.pyplot as plt
# import numpy as np
# plt.imshow(x, cmap='hot', interpolation='nearest')
# plt.savefig("look.png")
# plt.close()

# cuteoff the matrix
x_small = x[300:350, 200:250]
non_zero = np.sum(x_small>0)
n_row, n_col = x.shape
mask = (x_small==0)

prob_dict = {
  "m1": 50,
  "m2": 50,
  "R": 5,
  "noise": 0.01,
  "pre_mask": mask
}

prob = matrix_discover(**prob_dict)
prob.x = (x_small - 2.5) / 5
prob.x_noise = (x_small - 2.5) / 5

script_file = 'expt_real_data_movie.py'

expt_utils.algos_real_data(
  prob, algos_dict, matrix_discover, T = 100,
  results_dir='results/movie_mtx/smaller_noise_0.01', script_file=script_file
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
