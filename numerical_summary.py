# Get numerical summary
import pickle
import numpy as np
from tabulate import tabulate
import os
from glob import glob
import shutil
import pandas as pd
PATH = "/Users/zipingxu/Desktop/Research/Discovery/Code/Code/selected_results/ALT"
files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.pkl'))]
multi_var = True
i = 0

percentage = np.array([0.2])
# t_var = 0
results = []
for file in files:
    f = open(file, 'rb')
    reg_list_total = pickle.load(f)
    f.close()
    n_var = len(reg_list_total)
    algos_dict = list(reg_list_total.keys())
    T = len(reg_list_total[algos_dict[0]][0])
    tmp_y = []
    tmp_ci = []
    for algo in algos_dict:
        y = np.mean(reg_list_total[algo], axis = 0)[:T]
        ci = np.std(reg_list_total[algo], axis = 0)[:T]
        tmp_y.append(y[(percentage * T).astype(int)])
        tmp_ci.append(ci[(percentage * T).astype(int)])
    tmp_y = np.array(tmp_y)
    tmp_ci = np.array(tmp_ci)
    shapes = tmp_y.shape
    tmp_y_flat = tmp_y.reshape((-1, ))
    tmp_ci_flat = tmp_ci.reshape((-1, ))
    res = np.array(["%0.2f (%0.2f)" % (tmp_y_flat[i], tmp_ci_flat[i]) for i in range(len(tmp_y_flat))])
    res = res.reshape(shapes)
    col_name = ["{:.0%}".format(v) for v in percentage]
    res = pd.DataFrame(res, index = algos_dict, columns = col_name)
    results.append(res)
    # print(tabulate(res, tablefmt = "latex", headers = col_name))

results = pd.concat(results, axis = 1)
print(tabulate(results, tablefmt = "latex"))
