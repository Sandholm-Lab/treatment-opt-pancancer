"""
This file contains code to evaluate the results of the search process
over multiple cells.
"""

import numpy as np
from src.util.store import load_data
from src.util.verify import row_to_treatment, row_to_sequential_treatment, verify_search_result, verify_sequential_search_result
from src.env.cell_lines import retrieve_lines

# NOTE: can we determine our own prefix -> if lambda would be still exponent

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def recover_numbers_from_list(s):
    l = np.array([float(x) for x in s[1:-1].replace('\n', '').split(" ") if x])
    return l

# -------------------------------------------------------------------
# Functions for retrieval
# -------------------------------------------------------------------

# evaluate single cell

def best_single_search_result(cell_line, path, prefix, lambd=0, max_dosage=8000, verification=False):
    """
    Function for retrieval of single-cell search results.
    """
    data = load_data(path, cell_line, prefix=prefix, format="csv")
    objective = np.inf

    for i in data[data['threshold'] == max_dosage].index:
        assert data["threshold"][i] == max_dosage, "Retrieval from pandas frame does not work as expected."
        temp_prolif = data["relative_proliferation"][i]
        temp_objective = temp_prolif + lambd * data["total_concentration"][i]
        if temp_objective <= objective:
            objective = temp_objective
            rel_prolif = temp_prolif
            concentration = data["total_concentration"][i]
            treatment = row_to_treatment(data.iloc[i])

    if verification:
        print("- Verifying search")
        verify_search_result([cell_line], treatment, concentration, [rel_prolif], objective, "avg", lambd)

    # treatment, concentration, relative proliferation, objective value
    return treatment, concentration, rel_prolif, objective


def best_multi_search_result(tissue, path, prefix, lambd=0, obj="avg", max_dosage=8000, verification=False):
    """
    Function for retrieval of multi-cell search results.
    """
    data = load_data(path, tissue, prefix=prefix, format="csv")
    objective = np.inf

    for i in data[data['threshold'] == max_dosage].index:
        assert data["threshold"][i] == max_dosage, "Retrieval from pandas frame did not work as expected."
        temp_prolif_list = recover_numbers_from_list(data["relative_proliferation"][i])
        assert len(temp_prolif_list) == len(retrieve_lines(tissue)), "Number of proliferation values is off."
        if obj == "avg":
            temp_prolif = np.average(temp_prolif_list)
        elif obj == "worst":
            temp_prolif = np.max(temp_prolif_list)
        else:
            raise ValueError("Specified objective is unknown.")
        temp_objective = temp_prolif + lambd * data["total_concentration"][i]
        if temp_objective <= objective:
            objective = temp_objective
            rel_prolif = temp_prolif
            prolif_list = temp_prolif_list
            concentration = data["total_concentration"][i]
            treatment = row_to_treatment(data.iloc[i])

    if verification:
        print("- Verifying search")
        verify_search_result(retrieve_lines(tissue), treatment, concentration, prolif_list, objective, obj, lambd)

    # treatment, concentration, relative proliferation, objective value
    return treatment, concentration, rel_prolif, objective

def best_interpolated_multi_search_result(tissue, path, prefix, n_steps, lambd=0, obj="avg", max_dosage=8000, verification=False):
    """
    This methods retrieves the best single-step treatment from the multi-cell experiments and interpolates it over multiple steps.
    """
    data = load_data(path, tissue, prefix=prefix, format="csv")
    objective = np.inf

    for i in data[data['threshold'] == max_dosage].index:
        assert data["threshold"][i] == max_dosage, "Retrieval from pandas frame did not work as expected."
        temp_prolif_list = recover_numbers_from_list(data["relative_proliferation"][i])
        assert len(temp_prolif_list) == len(retrieve_lines(tissue)), "Number of proliferation values is off."
        if obj == "avg":
            temp_prolif = np.average(temp_prolif_list ** n_steps)
        elif obj == "worst":
            temp_prolif = np.max(temp_prolif_list ** n_steps)
        else:
            raise ValueError("Specified objective is unknown.")
        temp_objective = temp_prolif + lambd * data["total_concentration"][i] * n_steps
        if temp_objective <= objective:
            objective = temp_objective
            rel_prolif = temp_prolif
            prolif_list = temp_prolif_list ** n_steps
            concentration = data["total_concentration"][i] * n_steps
            treatments = [row_to_treatment(data.iloc[i]) for _ in range(n_steps)]

    if verification:
        print("- Verifying search")
        verify_sequential_search_result(retrieve_lines(tissue), n_steps, treatments, concentration, prolif_list, objective, obj, lambd)

    # treatment, concentration, relative proliferation, objective value
    return treatments, concentration, rel_prolif, objective

def best_sequential_search_result(tissue, path, prefix, n_steps, lambd=0, obj="avg", max_dosage=8000, verification=False):
    """
    Function for retrieval of multi-cell search results.
    """
    data = load_data(path, tissue, prefix=prefix, format="csv")
    objective = np.inf

    for i in data[data['threshold'] == max_dosage].index:
        assert data["threshold"][i] == max_dosage, "Retrieval from pandas frame did not work as expected."
        temp_prolif_list = recover_numbers_from_list(data["relative_proliferation"][i])
        assert len(temp_prolif_list) == len(retrieve_lines(tissue)), "Number of proliferation values is off."
        if obj == "avg":
            temp_prolif = np.average(temp_prolif_list)
        elif obj == "worst":
            temp_prolif = np.max(temp_prolif_list)
        else:
            raise ValueError("Specified objective is unknown.")
        temp_objective = temp_prolif + lambd * data["total_concentration"][i]
        if temp_objective <= objective:
            objective = temp_objective
            rel_prolif = temp_prolif
            prolif_list = temp_prolif_list
            concentration = data["total_concentration"][i]
            treatments = row_to_sequential_treatment(data.iloc[i], n_steps)
        
    if verification:
        print("- Verifying search")
        verify_sequential_search_result(retrieve_lines(tissue), n_steps, treatments, concentration, prolif_list, objective, obj, lambd)

    # treatment, concentration, relative proliferation, objective value
    return treatments, concentration, rel_prolif, objective
