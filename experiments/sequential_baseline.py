"""
This script runs experiments for sequential experiments.
"""

import os,sys,inspect
import argparse
import csv
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.env.cell_lines import retrieve_lines
from src.env.thresholds import THRESHOLDS
from src.search.evaluator import Evaluator
from src.search.covariance_matrix_adaption import cma_es
from src.util.domain import retrieve_domain
from src.env.objectives import MultiAvgLinear, MultiWorstLinear
from src.util.store import initialize_sequential_result_dictionary, update_sequential_result_dictionary, store, load_data
import numpy as np

# -------------------------------------------------------------------
# Setup conditions for experiments
# -------------------------------------------------------------------

# cma-es configuration
MAX_ITER = 200
N_ENVS = 9
SCALE = "linear"
THRES = [8000]

# path for optimization results
PATH = "./artifacts/sequential_baseline/"

# store all experimental evaluations
STORE = False

# -------------------------------------------------------------------
# Run CMA for each cell line
# -------------------------------------------------------------------

# find better way to store results

def cma_experiment(tissue, n_steps, domain, objective, prefix, seed):
    print(tissue)
    print("-----------------------")
    cell_lines = retrieve_lines(tissue)
    res_dict = initialize_sequential_result_dictionary(n_steps)
    res_dict["threshold"] = []
    for T in THRES:
        conf = {
            "n_steps": n_steps,
            "cell_lines": cell_lines,
            "objective": objective,
            "max_dosage": T,
            "domain": domain, 
            "scale": SCALE
        }
        evaluator = Evaluator(conf, n_envs=N_ENVS, store=STORE, repeated=True) # NOTE: repeated causes the evaluator to use the same treatment at every step
        mu, obj, rel_prolif = cma_es(evaluator, domain, MAX_ITER, verbose=True, seed=seed)
        assert len(rel_prolif) == len(cell_lines), "Number of proliferations differs from number of cell lines."
        update_sequential_result_dictionary(res_dict, [np.concatenate([mu] * n_steps)], [rel_prolif], T, SCALE, n_steps)
        res_dict["threshold"].append(T)
        evaluator.terminate()
    store(res_dict, PATH, tissue, prefix, format="csv")

# -------------------------------------------------------------------
# Finished experiment
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Create baselines for tissue.')
    parser.add_argument("-t", '--tissue', metavar='tissue', type=str, required=True,
                        help='the name of the relevant tissue. \
                        Possible tissues are "breast", "intestine", "lung", "pancreas", "skin" and "initial".')

    parser.add_argument("-l", '--lambd', metavar='lambd', type=float, required=False, default=12345,
                        help='Specifies the exponent of the weighting parameter for the linear penalty function. \
                        If no value is specified the algorithm optimizes only for relative proliferation. \
                        If a value is specified lambda 10^l is used.')

    parser.add_argument("-d", '--domain', metavar='domain', type=str, required=True,
                    help='the domain for the optimization process. \
                    Possible domains are "simplex" and "cube".')

    parser.add_argument("-o", '--objective', metavar='objective', type=str, required=True,
                        help='Specifies how the objective function uses the proliferation vector of the population. \
                        Possible values are "avg" and "worst" which cause the algorithm to optimize on the average and worst \
                        value case proliferation. The default value of the flag is "avg".')

    parser.add_argument("-s", '--steps', metavar='steps', type=int, required=True,
                    help='Specifies the number of steps used for the treatment plan.')

    parser.add_argument("-r", '--random_seed', metavar='random_seed', type=int, required=True,
                        help='Seed for random number generator.')

    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    args = parser.parse_args()
    seed = args.random_seed
    lambd = args.lambd
    domain = retrieve_domain(args.domain, dim=7, n_steps=1, seed=seed) # we repeat the treatment in the evaluator

    if lambd == 12345:
        lambd = 0
        prefix = str(args.steps) + "step_" + args.objective + "_" + args.domain + "_" + "prolif" + "_cma_es" # create prefix here and then give it to function
    else: 
        prefix = str(args.steps) + "step_" + args.objective + "_" + args.domain + "_" + str(lambd).replace(".", "_") + "_cma_es" # create prefix here and then give it to function
        lambd = 10 ** lambd

    if args.objective == "avg":
        objective = MultiAvgLinear(lambd)
    elif args.objective == "worst":
        objective = MultiWorstLinear(lambd)
    else:
        raise ValueError("The specified objective type is unknown.")

    print("Tissue:", args.tissue)
    print("Domain:", type(domain))
    print("objective:", args.objective)
    print("Prefix:", prefix)
    print("Lambda:", lambd)
    print("Steps:", args.steps)
    print("")

    print("Running optimization...")
    cma_experiment(args.tissue, args.steps, domain, objective, prefix, seed)
    print("Completed optimization.")

    print("\n----------------------------------------")
    print("Stored results successfully.")

if __name__ == '__main__':
    main()
