"""
This file contains simple function to verify our findings from the search process.
"""

from src.reference_simulator.simulator import Simulator
from src.env.drugs import DRUGS, empty_treatment
import numpy as np

EPS = 1e-4


def row_to_treatment(description):
    treatment = empty_treatment()
    for d in DRUGS:
        treatment[d] = description[d]
    return treatment

def row_to_sequential_treatment(description, n_steps):
    treatments = []
    for i in range(n_steps): # add indexing
        pre = "t" + str(i + 1) + "_"
        t = empty_treatment()
        for d in DRUGS:
            t[d] = description[pre + d]
        treatments.append(t)
    return treatments

def verify_search_result(cell_lines, treatment, dosage, prolifs, obj_val, obj_type, lambd):

    # check dosage value
    ref_dosage = 0
    for k in treatment:
        ref_dosage += treatment[k]
    assert np.abs(ref_dosage - dosage) < EPS, "Total concentration in record is off."

    # check proliferation values
    ref_prolifs = []
    for i, line in enumerate(cell_lines):
        simulator = Simulator()
        simulator.initialize(line)
        ref_prolifs.append(simulator.apply_treatment(treatment))
        assert np.abs(ref_prolifs[i] - prolifs[i]) < EPS, "Proliferation value in record is off."

    # check objective value
    if obj_type == "avg":
        ref_obj = np.average(ref_prolifs) + lambd * ref_dosage
    elif obj_type == "worst":
        ref_obj = np.max(ref_prolifs) + lambd * ref_dosage
    else:
        raise ValueError("Given objective is unknown.")
    assert np.abs(ref_obj - obj_val) < EPS, "Objective value in record is off."


def verify_sequential_search_result(cell_lines, n_steps, treatments, dosage, prolifs, obj_val, obj_type, lambd):

    # check dosage value
    ref_dosage = 0
    for t in treatments:
        for k in t:
            ref_dosage += t[k]
    assert np.abs(ref_dosage - dosage) < EPS, "Total concentration in record is off."

    # check proliferation values
    ref_prolifs = []
    for i, line in enumerate(cell_lines):
        simulator = Simulator()
        simulator.initialize(line)
        for j in range(n_steps):
            p = simulator.apply_treatment(treatments[j])
        ref_prolifs.append(p)
        assert np.abs(ref_prolifs[i] - prolifs[i]) < EPS, "Proliferation value in record is off."

    # check objective value
    if obj_type == "avg":
        ref_obj = np.average(ref_prolifs) + lambd * ref_dosage
    elif obj_type == "worst":
        ref_obj = np.max(ref_prolifs) + lambd * ref_dosage
    else:
        raise ValueError("Given objective is unknown.")
    assert np.abs(ref_obj - obj_val) < EPS, "Objective value in record is off."
