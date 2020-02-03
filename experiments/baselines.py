"""
This script runs experiments for the baseline treatments.
"""

import os,sys,inspect
import argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.baseline.generate import single_drug_baseline, two_drug_baseline
from src.env.cell_lines import retrieve_lines

# -------------------------------------------------------------------
# Setup conditions for experiments
# -------------------------------------------------------------------

# maximum doage for each experiment
MAX_DOSAGE = 8000

# granularity of sampling
STEP_SIZE = 1

# number of threads used for generation
WORKERS = 16

# perform two drug baseline
DUAL_BASELINE = True

# -------------------------------------------------------------------
# Single drug baseline
# -------------------------------------------------------------------

def single_baseline(cell_lines):
    for cell_line in cell_lines:
        print(cell_line)
        print("-----------------------")
        path = "./artifacts/baselines/" + cell_line + "_baseline.pkl"
        if os.path.isfile(path):
            print ("Single drug baseline data already exists for " + cell_line + ".")
        else:
            single_drug_baseline(cell_line, max_concentration=MAX_DOSAGE, step_size=STEP_SIZE, workers=WORKERS)
   
# -------------------------------------------------------------------
# Two drug baseline
# -------------------------------------------------------------------

def two_baseline(cell_lines):
    for cell_line in cell_lines:
        print(cell_line)
        print("-----------------------")
        path = "./artifacts/baselines/" + cell_line + "_dual.pkl"
        if os.path.isfile(path):
            print ("Two drug baseline data already exists for " + cell_line + ".")
        else:
            two_drug_baseline(cell_line, max_concentration=MAX_DOSAGE, step_size=STEP_SIZE, workers=WORKERS)

# -------------------------------------------------------------------
# Finished experiment
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Create baselines for tissue.')
    parser.add_argument("-t", '--tissue', metavar='tissue', type=str, required=True,
                        help='the name of the relevant tissue. \
                        Possible tissues are "breast", "intestine", "lung", "pancreas", "skin" and "initial".')

    args = parser.parse_args()
    cell_lines = retrieve_lines(args.tissue)

    single_baseline(cell_lines)
    if DUAL_BASELINE:
        two_baseline(cell_lines)

    print("----------------------------------------")
    print("Completed experimentation and stored baseline data successfully.")

if __name__ == '__main__':
    main()
