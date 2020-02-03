"""
This file contains code to visualize the results for the baseline
treatments. It can be used for single and two drug treatments.
"""

import pandas as pd
import matplotlib.pyplot as plt
from src.env.drugs import DRUGS

RATIOS = [x * 10 for x in range(11)] # 10% steps


def plot_single_drug_treatments(lines, max_dosage=8000):
    _, ax = plt.subplots(len(DRUGS),  figsize=(6,15)) 
    dfs = [pd.read_pickle("./artifacts/baselines/" + l + "_baseline.pkl") for l in lines]
    dfs = [df.loc[df['concentration'] <= max_dosage] for df in dfs]

    for i, d in enumerate(DRUGS):
        for j, df in enumerate(dfs):
            # add legend with drug names
            ax[i].plot(df['concentration'], df[d], label=lines[j])
        ax[i].set_title(d)
        ax[i].set_xscale('log', basex=2)
        ax[i].set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax[i].set_ylim([-0.1, 1.1])
        ax[i].legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('./artifacts/evaluation/baseline.png')                                                            


def plot_two_drug_treatments(cell_line, max_dosage=8000):
    _ = plt.figure(figsize=(5, 3))
    ax = plt.axes()
    data = pd.read_pickle("./artifacts/baselines/" + cell_line + "_dual.pkl")
    data = data.loc[data['concentration'] <= max_dosage]
    for r in RATIOS:
        ax.plot(data['concentration'], data[r], color="royalblue", alpha=(0.5 * (r / 200)))
    ax.set_title(cell_line+ ": PD0325901 + PLX-4720")
    ax.set_xscale('log', basex=2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim([-0.1, 1.1])
    plt.tight_layout()    
    plt.savefig("./artifacts/evaluation/" + cell_line + "_dual.png")                                                            
