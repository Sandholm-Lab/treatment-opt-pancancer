"""
This file contains code to create a dataframe containing the results for the single drug baseline
treatment. It uses parallelization to compute the table efficiently.
"""

import pandas as pd
from multiprocessing import Pool
from src.reference_simulator.simulator import Simulator
from src.env.drugs import DRUGS, empty_treatment

# used to distribute to the jobs
BATCH_NUMBER = 20

# used for the two drug treatment
RATIOS = [x * 5 for x in range(21)] # 5% steps

# -------------------------------------------------------------------
# Code for single drug baseline
# -------------------------------------------------------------------

def experiment_batch(arg):
    simulator = Simulator()
    res = []
    for con in arg["concentations"]:
        treatment = empty_treatment()
        treatment[arg["drug"]] = con

        simulator.initialize(arg["cell_line"])
        rel_proliferation = simulator.apply_treatment(treatment)
        simulator.initialized = False

        res.append(rel_proliferation)
    return res

def single_drug_baseline(cell_line, max_concentration=8000, step_size=10, workers=8):
    assert max_concentration % step_size == 0, "max_concentration needs to be a multiple of the step size."

    steps = max_concentration // step_size + 1
    concentrations = [step_size * i for i in range(steps)]
    n = steps // BATCH_NUMBER + 1 # causes slight imbalance, but should be ok for larger experiments
    batches = [concentrations[i * n : (i + 1) * n] for i in range(BATCH_NUMBER)]

    print("Preparing jobs...")
    jobs = []
    for drug in DRUGS:
        for batch in batches:
            job = {
                'drug': drug,
                'cell_line': cell_line,
                'concentations': batch,
            }
            jobs.append(job)

    print("Running experiments...")
    worker_pool = Pool(processes=workers)
    data = worker_pool.map(experiment_batch, jobs)
    worker_pool.close()

    print("Preparing results...")
    res_dict = {}
    res_dict['concentration'] = concentrations
    for drug in DRUGS:
        res_dict[drug] = []

    for i, res in enumerate(data):
        res_dict[jobs[i]['drug']].extend(res)

    frame = pd.DataFrame.from_dict(res_dict)
    print(frame.head)

    print("Storing results...")
    frame.to_pickle("./artifacts/baselines/" + cell_line + "_baseline.pkl")

    print("Completed baseline successfully. (:")


# -------------------------------------------------------------------
# Code for dual drug baseline
# -------------------------------------------------------------------

def dual_drug_batch(arg):
    simulator = Simulator()
    res = []
    for con in arg["concentations"]:
        treatment = empty_treatment()
        treatment['PD0325901'] = (1 - (arg["ratio"] / 100.0)) * con
        treatment['PLX-4720'] = (arg["ratio"] / 100.0) * con

        simulator.initialize(arg["cell_line"])
        rel_proliferation = simulator.apply_treatment(treatment)
        simulator.initialized = False

        res.append(rel_proliferation)
    return res

def two_drug_baseline(cell_line, max_concentration=8000, step_size=10, workers=8):
    assert max_concentration % step_size == 0, "max_concentration needs to be a multiple of the step size."

    steps = max_concentration // step_size + 1
    concentrations = [step_size * i for i in range(steps)]
    n = steps // BATCH_NUMBER + 1 # causes slight imbalance, but should be ok for larger experiments
    batches = [concentrations[i * n : (i + 1) * n] for i in range(BATCH_NUMBER)]

    print("Preparing jobs...")
    jobs = []
    for r in RATIOS:
        for batch in batches:
            job = {
                'ratio': r,
                'cell_line': cell_line,
                'concentations': batch,
            }
            jobs.append(job)

    print("Running experiments...")
    worker_pool = Pool(processes=workers)
    data = worker_pool.map(dual_drug_batch, jobs)
    worker_pool.close()

    print("Preparing results...")
    res_dict = {}
    res_dict['concentration'] = concentrations
    for r in RATIOS:
        res_dict[r] = []

    for i, res in enumerate(data):
        res_dict[jobs[i]["ratio"]].extend(res)

    frame = pd.DataFrame.from_dict(res_dict)
    print(frame.head)

    print("Storing results...")
    frame.to_pickle("./artifacts/baselines/" + cell_line + "_dual.pkl")

    print("Completed baseline successfully. (:")
