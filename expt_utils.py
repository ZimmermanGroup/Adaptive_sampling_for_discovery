import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg
from scipy.stats import linregress
import pdb
from utils import *
from datetime import datetime
import importlib
import pickle
import os
from tqdm import trange
from copy import deepcopy


def dump_script(
    dirname, script_file, dest=None, timestamp=None, file_list=None):
    import glob, os, shutil, sys
    from datetime import datetime

    if dest is None:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(
            dirname, 'script_{}'.format(timestamp))
    os.mkdir(dest)

    print('copying files to {}'.format(dest))
    if file_list is None:
        file_list = glob.glob("*.py")
    for file in file_list:
        print('copying {}'.format(file))
        shutil.copy2(file, dest)
    print('copying {}'.format(script_file))
    shutil.copy2(script_file, dest)

    with open(os.path.join(dest, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

def save_output(output, timestamp, dir_name=None):
    if dir_name == None:
        dir_name = '.'
    f = open('{}/output_{}.pkl'.format(
    dir_name, timestamp), 'wb')
    pickle.dump(output,f)
    f.close()

def algos_vs_var_metrics(
    prob_dict, algos_dict, prob_function, T,
    results_dir = None, script_file=None, xlims = None, ylims = None, load = None, log_y = False
    ):
    list_num = [isinstance(prob_dict[k], list) for k in prob_dict]
    if sum(list_num) > 1:
        raise Exception("More than one variables!")
    variable = np.array(list(prob_dict.keys()))[list_num][0]

    N_trials = 1
    for algo in algos_dict:
        algo_N_trials = algos_dict[algo][2]
    if algo_N_trials < 1:
        raise ValueError
    N_trials = max(algo_N_trials, N_trials)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = 'results' if results_dir is None else results_dir
    expt_name = '{}_vs_{}'.format(
        '-'.join(algos_dict.keys()), timestamp)
    dir_name = os.path.join(results_dir, expt_name)
    os.makedirs(dir_name, exist_ok=True)
    dump_script(dir_name, script_file, timestamp=timestamp, file_list=[])

    end_list = {algo:[] for algo in algos_dict}
    reg_list_total = []
    if load == None:
        for t_var in prob_dict[variable]:
            reg_list = {algo:[] for algo in algos_dict}
            _end_list = {algo:[] for algo in algos_dict}
            t_prob_dict = deepcopy(prob_dict)
            t_prob_dict[variable] = t_var
            for trial_idx in trange(N_trials):
                prob = prob_function(**t_prob_dict)
                # print(prob.x)
                for algo, algo_setup in algos_dict.items():
                    algo_func, algo_params, algo_N_trials, _, _ = algo_setup
                    # print(algo_func)
                    if not trial_idx < algo_N_trials:
                        continue
                    solver = algo_func(prob, **algo_params)
                    reg, _ = solver.run(T)
                    # print(prob.rewards)
                    reg_list[algo].append(reg)
                    _end_list[algo].append(reg[-1]) # ultimate regret
            for algo in algos_dict:
                algo_label = algos_dict[algo][3]
                algo_style = algos_dict[algo][4]
                plt.plot(range(T), np.mean(reg_list[algo], axis = 0),
                          algo_style, label=algo_label)
                if log_y:
                    plt.yscale('log')
            if xlims is not None:
                plt.xlim(xlims)
            if ylims is not None:
                plt.ylim(ylims)
            plt.legend()
            plot_file_name = expt_name

            # plt.title(plt_titles[metric])
            plt.xlabel("T")
            plt.ylabel("Regrets")
            # plt.show()
            plt.savefig(os.path.join(dir_name, plot_file_name + "_" + str(t_var) +'_labeled.png'))
            plt.savefig(os.path.join(dir_name, plot_file_name + "_" + str(t_var) +'_labeled.pdf'))
            plt.close()
            for algo in algos_dict:
                end_list[algo].append(_end_list[algo])
            reg_list_total.append(reg_list)
    else:
        f = open(load, 'rb')
        reg_list_total = pickle.load(f)
        f.close()
        # pdb.set_trace()
        for t_var in range(len(prob_dict[variable])):
            for algo, algo_setup in algos_dict.items():
                tmp = [reg_list_total[t_var][algo][trial_idx][T-1] for trial_idx in trange(N_trials)]
                end_list[algo].append(tmp)
        for t_var in range(len(prob_dict[variable])):
            for algo in algos_dict:
                algo_label = algos_dict[algo][3]
                algo_style = algos_dict[algo][4]
                y = np.mean(reg_list_total[t_var][algo], axis = 0)[:T]
                ci = np.std(reg_list_total[t_var][algo], axis = 0)[:T]
                plt.plot(range(T), y,
                          "-" + algo_style, label=algo_label)
                plt.fill_between(range(T), (y-ci), (y+ci), color = algo_style, alpha=.1)
                if log_y:
                    plt.yscale('log')
            if xlims is not None:
                plt.xlim(xlims)
            if ylims is not None:
                plt.ylim(ylims)
            plt.legend()
            plot_file_name = expt_name

            plt.xlabel("T")
            plt.ylabel("Regrets")
            plt.savefig(os.path.join(dir_name, plot_file_name + "_" + str(t_var) +'_labeled.png'))
            plt.savefig(os.path.join(dir_name, plot_file_name + "_" + str(t_var) +'_labeled.pdf'))
            plt.close()

    save_output(reg_list_total, timestamp, dir_name)
    # plot summary
    # pdb.set_trace()
    # a = 1
    for algo in algos_dict:
        algo_label = algos_dict[algo][3]
        algo_style = algos_dict[algo][4]
        # pdb.set_trace()
        y = np.mean(end_list[algo], axis = 1)
        ci = np.std(end_list[algo], axis = 1)
        plt.plot(prob_dict[variable], y,
                    "-s" + algo_style, label=algo_label)
        plt.fill_between(prob_dict[variable], (y-ci), (y+ci), color = algo_style, alpha=.1)
        if log_y:
            plt.yscale('log')
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    plt.legend()
    plot_file_name = expt_name

    # plt.title(plt_titles[metric])
    plt.xlabel(variable)
    plt.ylabel("%d-step Regrets"%(T))
    # plt.show()
    plt.savefig(os.path.join(dir_name, plot_file_name  +'_summarized_labeled.png'))
    plt.savefig(os.path.join(dir_name, plot_file_name  +'_summarized_labeled.pdf'))
    plt.close()


def algos_metrics(
    prob_dict, algos_dict, prob_function, T,
    results_dir = None, script_file=None, xlims = None, ylims = None, load = None, log_y = False
    ):
    N_trials = 1
    for algo in algos_dict:
        algo_N_trials = algos_dict[algo][2]
    if algo_N_trials < 1:
        raise ValueError
    N_trials = max(algo_N_trials, N_trials)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = 'results' if results_dir is None else results_dir
    expt_name = '{}_vs_{}'.format(
        '-'.join(algos_dict.keys()), timestamp)
    dir_name = os.path.join(results_dir, expt_name)
    os.makedirs(dir_name, exist_ok=True)
    dump_script(dir_name, script_file, timestamp=timestamp, file_list=[])
    reg_list = {algo:[] for algo in algos_dict}
    if load == None:
        for trial_idx in trange(N_trials):
            prob = prob_function(**prob_dict)
            # print(prob.x)
            for algo, algo_setup in algos_dict.items():
                algo_func, algo_params, algo_N_trials, _, _ = algo_setup
                # print(algo_func)
                if not trial_idx < algo_N_trials:
                    continue
                solver = algo_func(prob, **algo_params)
                reg, _ = solver.run(T)
                # print(prob.rewards)
                reg_list[algo].append(reg)
    else:
        f = open(load, 'rb')
        reg_list = pickle.load(f)
        f.close()
    save_output(reg_list, timestamp, dir_name)
    for algo in algos_dict:
        algo_label = algos_dict[algo][3]
        algo_style = algos_dict[algo][4]
        y = np.mean(reg_list[algo], axis = 0)[:T]
        ci = np.std(reg_list[algo], axis = 0)[:T]
        plt.plot(range(T), y,
                  algo_style, label=algo_label)
        plt.fill_between(range(T), (y-ci), (y+ci), color = algo_style, alpha=.1)
        if log_y:
            plt.yscale('log')
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    plt.legend()
    plot_file_name = expt_name

    # plt.title(plt_titles[metric])
    plt.xlabel("T")
    plt.ylabel("Regrets")
    # plt.show()
    plt.savefig(os.path.join(dir_name, plot_file_name+'_labeled.png'))
    plt.savefig(os.path.join(dir_name, plot_file_name+'_labeled.pdf'))
    plt.close()


def algos_real_data(
    prob, algos_dict, prob_function, T,
    results_dir = None, script_file=None, xlims = None, ylims = None, load = None, log_y = False
    ):
    N_trials = 1
    for algo in algos_dict:
        algo_N_trials = algos_dict[algo][2]
    if algo_N_trials < 1:
        raise ValueError
    N_trials = max(algo_N_trials, N_trials)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = 'results' if results_dir is None else results_dir
    expt_name = '{}_vs_{}'.format(
        '-'.join(algos_dict.keys()), timestamp)
    dir_name = os.path.join(results_dir, expt_name)
    os.makedirs(dir_name, exist_ok=True)
    dump_script(dir_name, script_file, timestamp=timestamp, file_list=[])
    reg_list = {algo:[] for algo in algos_dict}
    if load == None:
        for trial_idx in trange(N_trials):
            t_prob = deepcopy(prob)
            # print(prob.x)
            for algo, algo_setup in algos_dict.items():
                algo_func, algo_params, algo_N_trials, _, _ = algo_setup
                # print(algo_func)
                if not trial_idx < algo_N_trials:
                    continue
                solver = algo_func(t_prob, **algo_params)
                reg, _ = solver.run(T)
                # print(prob.rewards)
                reg_list[algo].append(reg)
    else:
        f = open(load, 'rb')
        reg_list = pickle.load(f)
        f.close()
    save_output(reg_list, timestamp, dir_name)
    # output, timestamp, dir_name=None
    for algo in algos_dict:
        algo_label = algos_dict[algo][3]
        algo_style = algos_dict[algo][4]
        y = np.mean(reg_list[algo], axis = 0)[:T]
        ci = np.std(reg_list[algo], axis = 0)[:T]
        plt.plot(range(T), y,
                  "-"+algo_style, label=algo_label)
        plt.fill_between(range(T), (y-ci), (y+ci), color = algo_style, alpha=.1)
        if log_y:
            plt.yscale('log')
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    plt.legend()
    plot_file_name = expt_name

    # plt.title(plt_titles[metric])
    plt.xlabel("T")
    plt.ylabel("Regrets")
    # plt.show()
    plt.savefig(os.path.join(dir_name, plot_file_name+'_labeled.png'))
    plt.savefig(os.path.join(dir_name, plot_file_name+'_labeled.pdf'))
    plt.close()
    # generate numerical summary
