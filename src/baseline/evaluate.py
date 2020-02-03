"""
This file contains code to evaluate the results for the baseline
treatments over multiple cells. It can be used for single and two drug treatments.
"""

import numpy as np
import pandas as pd
from src.env.drugs import DRUGS, single_treatment, dual_treatment
from src.util.verify import verify_search_result, verify_sequential_search_result

RATIOS = [x * 5 for x in range(21)] # 5% steps

# -------------------------------------------------------------------
# Create combined dataframe describing the population
# -------------------------------------------------------------------

def build_combined_single_frame(cell_lines, drugs=DRUGS, max_dosage=8000, path="./artifacts/baselines/"):
    data = {}
    for line in cell_lines:
        data[line] = pd.read_pickle(path + line + "_baseline.pkl")

    combined_data = pd.read_pickle(path + cell_lines[0] + "_baseline.pkl")
    for d in drugs:
        vals = np.concatenate([np.vstack(data[k][d].values) for k in data], axis=1)
        vals = [vals[i] for i in range(len(vals))]
        combined_data[d] = vals
    return combined_data


def build_combined_dual_frame(cell_lines, ratios=RATIOS, max_dosage=8000, path="./artifacts/baselines/"):    
    data = {}
    for line in cell_lines:
        data[line] = pd.read_pickle(path + line + "_dual.pkl")

    combined_data = pd.read_pickle(path + cell_lines[0] + "_dual.pkl")
    for r in ratios:
        vals = np.concatenate([np.vstack(data[k][r].values) for k in data], axis=1)
        vals = [vals[i] for i in range(len(vals))]
        combined_data[r] = vals
    return combined_data


def evaluate_prolif_vector(prolifs, obj):
    if obj == "avg":
        p = np.average(prolifs)
    elif obj == "worst":
        p = np.max(prolifs)
    else:
        raise ValueError("Specified objective is unknown.")
    return p

# -------------------------------------------------------------------
# Code for data retrieval
# -------------------------------------------------------------------

def best_single_treatment(cell_lines, lambd=0, obj="avg", drugs=DRUGS, max_dosage=8000, path="./artifacts/baselines/", verification=False, comb_data=None):
    """
    This function uses the linear objective function in order to determine the best treatment. 
    After that it returns the drug name, concentration, proliferation and objective value.
    A choice of lambd = 0 implies that objective = prolif.
    """
    assert max_dosage <= 8000, "Maximum concentration needs to be less than 8000"
    if comb_data is None:
        data = build_combined_single_frame(cell_lines)
    else:
        data = comb_data.copy()

    # first analyze objective values
    for d in DRUGS: # NOTE: We intend this to be DRUGS
        column = np.array([a for a in data[d].values])
        if obj == "avg":
            data[d] = np.average(column, axis=1) + lambd * data['concentration'].values 
        elif obj == "worst":
            data[d] = np.max(column, axis=1) + lambd * data['concentration'].values 
        else:
            raise ValueError("Specified objective is unknown.")
    ids = data.loc[data['concentration'] <= max_dosage].idxmin(axis=0)
    objectives = {}
    for d in drugs: # determine best objective value
        objectives[d] = data[d][ids[d]]

    # reload to restore proliferation values
    if comb_data is None:
        data = build_combined_single_frame(cell_lines)
    else:
        data = comb_data.copy()
    best_drug = drugs[0]
    for d in drugs:
        if objectives[d] <= objectives[best_drug]:
            best_drug = d

    concentration = data['concentration'][ids[best_drug]]
    rel_prolif = data[best_drug][ids[best_drug]]
    objective = objectives[best_drug]

    if verification:
        print("- Verifying single")
        treatment = single_treatment(best_drug, concentration)
        verify_search_result(cell_lines, treatment, concentration, rel_prolif, objective, obj, lambd)

    rel_prolif = evaluate_prolif_vector(rel_prolif, obj)
    # drug name, concentration, relative proliferation, objective value
    return best_drug, concentration, rel_prolif, objective

def best_single_treatment_by_dosage(cell_lines, obj="avg", drugs=DRUGS, path="./artifacts/baselines/", comb_data=None):
    """
    This function uses the linear objective function in order to determine the best treatment. 
    After that it returns the drug name, concentration, proliferation and objective value.
    A choice of lambd = 0 implies that objective = prolif.
    """
    if comb_data is None:
        data = build_combined_single_frame(cell_lines)
    else:
        data = comb_data.copy()

    # first analyze objective values
    best_drugs = ["none" for i in range(len(data['concentration']))]
    best_prolifs =  [777 for i in range(len(data['concentration']))]
    for d in DRUGS: # NOTE: We intend this to be DRUGS
        column = np.array([a for a in data[d].values])
        if obj == "avg":
            data[d] = np.average(column, axis=1)
        elif obj == "worst":
            data[d] = np.max(column, axis=1)
        else:
            raise ValueError("Specified objective is unknown.")
        for i in range(len(data['concentration'])):
            if data[d][i] <= best_prolifs[i]:
                best_drugs[i] = d
                best_prolifs[i] =  data[d][i]

    # drug name, concentration, relative proliferation, objective value
    return best_drugs, data['concentration'], best_prolifs


def best_sequential_single_treatment(cell_lines, n_steps, lambd=0, obj="avg", drugs=DRUGS, max_dosage=8000, path="./artifacts/baselines/", verification=False,  comb_data=None):
    """
    This function uses the linear objective function in order to determine the best treatment.
    It interpolates the proliferation values from the single step experiments. 
    After that it returns the drug name, concentration, proliferation and objective value.
    A choice of lambd = 0 implies that objective = prolif.
    """
    assert max_dosage <= 8000, "Maximum concentration needs to be less than 8000"
    if comb_data is None:
        data = build_combined_single_frame(cell_lines)
    else:
        data = comb_data.copy()
        
    # first analyze objective values based on interpolation of the single step results
    for d in DRUGS: # NOTE: We intend this to be DRUGS
        column = np.array([a for a in data[d].values])
        if obj == "avg":
            data[d] = np.average(column ** n_steps, axis=1) + lambd * data['concentration'].values * n_steps
        elif obj == "worst":
            data[d] = np.max(column ** n_steps, axis=1) + lambd * data['concentration'].values * n_steps
        else:
            raise ValueError("Specified objective is unknown.")
    ids = data.loc[data['concentration'] <= max_dosage].idxmin(axis=0)
    objectives = {}
    for d in drugs: # determine best objective value
        objectives[d] = data[d][ids[d]]

    # reload to restore proliferation values
    if comb_data is None:
        data = build_combined_single_frame(cell_lines)
    else:
        data = comb_data.copy()

    best_drug = drugs[0]
    for d in drugs:
        if objectives[d] <= objectives[best_drug]:
            best_drug = d

    concentration = data['concentration'][ids[best_drug]] # NOTE: be careful with this
    rel_prolif = data[best_drug][ids[best_drug]] ** n_steps
    objective = objectives[best_drug]

    if verification:
        print("- Verifying single")
        treatments = [single_treatment(best_drug, concentration) for i in range(n_steps)]
        verify_sequential_search_result(cell_lines, n_steps, treatments, concentration * n_steps, rel_prolif, objective, obj, lambd)

    rel_prolif = evaluate_prolif_vector(rel_prolif, obj)
    # drug name, concentration, relative proliferation, objective value
    return best_drug, concentration * n_steps, rel_prolif, objective


def best_dual_treatment(cell_lines, lambd=0, obj="avg", ratios=RATIOS, max_dosage=8000, path="./artifacts/baselines/", verification=False,  comb_data=None):
    """
    This function uses the linear objective function in order to determine the best treatment. 
    After that it returns the drug name, concentration, proliferation and objective value.
    A choice of lambd = 0 implies that objective = prolif.
    """
    assert max_dosage <= 8000, "Maximum concentration needs to be less than 8000"
    if comb_data is None:
        data = build_combined_dual_frame(cell_lines)
    else:
        data = comb_data.copy()

    # first analyze objective values
    for r in RATIOS: # NOTE: We intend this to be RATIOS
        column = np.array([a for a in data[r].values])
        if obj == "avg":
            data[r] = np.average(column, axis=1) + lambd * data['concentration'].values 
        elif obj == "worst":
            data[r] = np.max(column, axis=1) + lambd * data['concentration'].values 
        else:
            raise ValueError("Specified objective is unknown.")
    ids = data.loc[data['concentration'] <= max_dosage].idxmin(axis=0)
    objectives = {}
    for r in ratios: # determine best objective value
        objectives[r] = data[r][ids[r]]

    # reload to restore proliferation values
    if comb_data is None:
        data = build_combined_dual_frame(cell_lines)
    else:
        data = comb_data.copy()
    best_r = ratios[0]
    for r in ratios:
        if objectives[r] <= objectives[best_r]:
            best_r = r

    concentration = data['concentration'][ids[best_r]]
    rel_prolif = data[best_r][ids[best_r]]
    objective = objectives[best_r]

    if verification:
        print("- Verifying dual")
        treatment = dual_treatment(int(best_r), concentration)
        verify_search_result(cell_lines, treatment, concentration, rel_prolif, objective, obj, lambd)

    rel_prolif = evaluate_prolif_vector(rel_prolif, obj)
    # drug name, concentration, relative proliferation, objective value
    return best_r, concentration, rel_prolif, objective


def best_sequential_dual_treatment(cell_lines, n_steps, lambd=0, obj="avg", ratios=RATIOS, max_dosage=8000, path="./artifacts/baselines/", verification=False,  comb_data=None):
    """
    This function uses the linear objective function in order to determine the best treatment. 
    After that it returns the drug name, concentration, proliferation and objective value.
    A choice of lambd = 0 implies that objective = prolif.
    """
    assert max_dosage <= 8000, "Maximum concentration needs to be less than 8000"
    if comb_data is None:
        data = build_combined_dual_frame(cell_lines)
    else:
        data = comb_data.copy()
    # first analyze objective values
    for r in RATIOS: # NOTE: We intend this to be RATIOS
        column = np.array([a for a in data[r].values])
        if obj == "avg":
            data[r] = np.average(column ** n_steps, axis=1) + lambd * data['concentration'].values * n_steps
        elif obj == "worst":
            data[r] = np.max(column ** n_steps, axis=1) + lambd * data['concentration'].values * n_steps 
        else:
            raise ValueError("Specified objective is unknown.")
    ids = data.loc[data['concentration'] <= max_dosage].idxmin(axis=0)
    objectives = {}
    for r in ratios: # determine best objective value
        objectives[r] = data[r][ids[r]]

    # reload to restore proliferation values
    if comb_data is None:
        data = build_combined_dual_frame(cell_lines)
    else:
        data = comb_data.copy()
    best_r = ratios[0]
    for r in ratios:
        if objectives[r] <= objectives[best_r]:
            best_r = r

    concentration = data['concentration'][ids[best_r]] # NOTE: be careful with this
    rel_prolif = data[best_r][ids[best_r]] ** n_steps
    objective = objectives[best_r]

    if verification:
        print("- Verifying dual")
        treatments = [dual_treatment(int(best_r), concentration) for i in range(n_steps)]
        verify_sequential_search_result(cell_lines, n_steps, treatments, concentration * n_steps, rel_prolif, objective, obj, lambd)

    rel_prolif = evaluate_prolif_vector(rel_prolif, obj)
    # drug name, concentration, relative proliferation, objective value
    return best_r, concentration * n_steps, rel_prolif, objective # NOTE: Be careful with concentration


def best_dual_treatment_by_dosage(cell_lines, obj="avg", ratios=RATIOS, path="./artifacts/baselines/", comb_data=None):
    """
    This function uses the linear objective function in order to determine the best treatment. 
    After that it returns the drug name, concentration, proliferation and objective value.
    A choice of lambd = 0 implies that objective = prolif.
    """
    if comb_data is None:
        data = build_combined_dual_frame(cell_lines)
    else:
        data = comb_data.copy()

    # first analyze objective values
    best_ratios = ["none" for i in range(len(data['concentration']))]
    best_prolifs =  [777 for i in range(len(data['concentration']))]
    for r in RATIOS: # NOTE: We intend this to be RATIOS
        column = np.array([a for a in data[r].values])
        if obj == "avg":
            data[r] = np.average(column, axis=1)
        elif obj == "worst":
            data[r] = np.max(column, axis=1)
        else:
            raise ValueError("Specified objective is unknown.")
        for i in range(len(data['concentration'])):
            if data[r][i] <= best_prolifs[i]:
                best_ratios[i] = r
                best_prolifs[i] =  data[r][i]

    # drug name, concentration, relative proliferation, objective value
    return best_ratios, data['concentration'], best_prolifs
