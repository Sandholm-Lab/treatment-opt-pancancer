"""
A simple helper functions to convert the result to a dataframe and store them.
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import pandas as pd
import datetime
from src.util.prepare_dict import prepare_dict


def initialize_result_dictionary():
    res_dict = {
        "relative_proliferation": [],
        "total_concentration": [],
        "PD0325901": [],
        "PLX-4720": [],
        "Selumetinib": [],
        "Lapatinib": [],
        "Erlotinib": [],
        "CHIR-265": [],
        "Vandetanib": [],
    }
    return res_dict

def update_result_dictionary(res_dict, xs, ys, max_dosage, scale):
    for i, x in enumerate(xs):
        res_dict["relative_proliferation"].append(ys[i])
        action_dict = prepare_dict(x, max_dosage=max_dosage, scale=scale)
        total_con = 0
        for k in action_dict:
            total_con += action_dict[k]
            res_dict[k].append(action_dict[k])
        res_dict["total_concentration"].append(total_con)

def initialize_sequential_result_dictionary(n_steps):
    res_dict = {}
    res_dict["relative_proliferation"] = [] 
    res_dict["total_concentration"] = []
    for i in range(1, n_steps + 1):
        pre = "t" + str(i) + "_"
        res_dict[pre + "PD0325901"] = []
        res_dict[pre + "PLX-4720"] = []
        res_dict[pre + "Selumetinib"] = []
        res_dict[pre + "Lapatinib"] = []
        res_dict[pre + "Erlotinib"] = []
        res_dict[pre + "CHIR-265"] = []
        res_dict[pre + "Vandetanib"] = []
    return res_dict

def update_sequential_result_dictionary(res_dict, xs, ys, max_dosage, scale, n_steps):
    for i, x in enumerate(xs):
        res_dict["relative_proliferation"].append(ys[i])
        total_con = 0
        for j in range(n_steps):
            pre = "t" + str(j + 1) + "_"
            action_dict = prepare_dict(x[j * 7: (j+1) * 7], max_dosage=max_dosage, scale=scale)
            for k in action_dict:
                total_con += action_dict[k]
                res_dict[pre + k].append(action_dict[k])
        res_dict["total_concentration"].append(total_con)

def store(res_dict, save_path, cell_line, method, verbose=True, format="pkl"):

    frame = pd.DataFrame.from_dict(res_dict)
    if verbose:
        print(frame.head)
        
    if not os.path.isdir(save_path + cell_line):
        os.mkdir(save_path + cell_line)

    now = datetime.datetime.now()
    time_string = str(now.day) + "_" + str(now.month) + "_" + str(now.year) + "_" \
                + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)

    if format == "pkl":
        file_path = cell_line + "/" + method + "_" + time_string + ".pkl"
        frame.to_pickle(save_path + file_path)
    elif format == "csv":
        file_path = cell_line + "/" + method + "_" + time_string + ".csv"
        frame.to_csv(save_path + file_path)
    else:
        raise ValueError("Unknown file format")

def load_data(path, cell_line, prefix="", format="pkl"):
    rel_files = [f for f in os.listdir(path + cell_line) if f.startswith(prefix)]
    if len(rel_files) == 0:
        raise ValueError("Could not retrieve data for given specification:\n Path:" + path \
            + cell_line + " Prefix: " + prefix) 
    dfs = []
    for f in rel_files:
        if format == "pkl":
            dfs.append(pd.read_pickle(path + cell_line + "/" + f))
        elif format == "csv":
            dfs.append(pd.read_csv(path + cell_line + "/" + f))
        else:
            raise ValueError("Unknown file format")
    data = pd.concat(dfs, ignore_index=True)
    return data
