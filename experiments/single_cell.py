"""
This script runs experiments for single cell-line experiments.
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
from src.env.objectives import SingleLinear
from src.util.store import initialize_result_dictionary, update_result_dictionary, store, load_data

# -------------------------------------------------------------------
# Setup conditions for experiments
# -------------------------------------------------------------------

# cma-es configuration
MAX_ITER = 200
N_ENVS = 9
SCALE = "linear"
THRES = [8000]

# path for optimization results
PATH = "./artifacts/single/"

# store all experimental evaluations
STORE = False

# -------------------------------------------------------------------
# Run CMA for each cell line
# -------------------------------------------------------------------

def cma_experiment(cell_line, domain, lambd, prefix, seed):
    print(cell_line)
    print("-----------------------")
    res_dict = initialize_result_dictionary()
    res_dict["threshold"] = []
    for T in THRES:
        conf = {
            "n_steps": 1,
            "cell_lines": [cell_line],
            "objective": SingleLinear(lambd),
            "max_dosage": T,
            "domain": domain, 
            "scale": SCALE
        }
        evaluator = Evaluator(conf, n_envs=N_ENVS, store=STORE)
        mu, obj, rel_prolif = cma_es(evaluator, domain, MAX_ITER, verbose=True, seed=seed)
        assert len(rel_prolif) == 1, "single cell experiment should only receive single return value"
        update_result_dictionary(res_dict, [mu], [rel_prolif[0]], T, SCALE)
        res_dict["threshold"].append(T)
        evaluator.terminate()
    store(res_dict, PATH, cell_line, prefix, format="csv")

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

    parser.add_argument("-r", '--random_seed', metavar='random_seed', type=int, required=True,
                        help='Seed for random number generator.')

    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    args = parser.parse_args()
    seed = args.random_seed
    cell_lines = retrieve_lines(args.tissue)
    lambd = args.lambd
    domain = retrieve_domain(args.domain, seed=seed)

    if lambd == 12345:
        lambd = 0
        prefix = args.domain + "_" + "prolif" + "_cma_es" # create prefix here and then give it to function
    else: 
        prefix = args.domain + "_" + str(lambd).replace(".", "_") + "_cma_es" # create prefix here and then give it to function
        lambd = 10 ** lambd
    
    print("Prefix:", prefix)
    print("Lambda:", lambd)
    print("")

    print("Running optimization...")
    for cell_line in cell_lines:
        cma_experiment(cell_line, domain, lambd, prefix, seed)
    print("Completed optimization.")

    print("\n----------------------------------------")
    print("Stored results successfully.")

if __name__ == '__main__':
    main()
