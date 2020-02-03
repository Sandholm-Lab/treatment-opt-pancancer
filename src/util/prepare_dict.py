"""
Some helper functions which allow to transform unit simplices and unit cubes into treatment
dictionaries which can be used as simulator input.
"""
import math
import numpy as np
from src.env.drugs import DRUGS

def splice_vector(mu, n_steps):
    "transforms a no pd vector into a full treatment vector"
    new_mu = []
    print("SLICING")
    for i in range(n_steps):
        new_mu += [float(0.0)]
        new_mu += [float(val) for val in mu[(i * 6):((i + 1) * 6)]]
    return new_mu

def from_log_scale(val, max_value):
    """
    NOTE: The current version of the code has a bug when using this version of log-scale
    because it is not able to use the maximum dosage when multiple drugs are used. At
    the moment only linear scale works reliably.
    """
    return ((max_value + 1) ** val) - 1

def to_log_scale(val, max_value):
    val = val + 1
    return math.log(val, (max_value + 1))

def from_linear_scale(val, max_value):
    return val * max_value

def to_linear_scale(val, max_value):
    return val / max_value

def prepare_dict(concentrations, max_dosage=8000, scale="linear"):
    """ Takes a numpy array and prepares it for the cell simulator.
    
    It transforms the [0, 1]-interval into a log- or linear-scale. If scale is picked to 
    bed real than it just compies the concentrations into a dictionary.

        Args:
            concentrations: A numpy array describing the concentrations for the treatment
                as a point in the unit simplex.
            max_dosage: A 1 concentration corresponds to this value.

        Returns:
            treatment: A dictionary for the simulator.
    """
    assert len(concentrations) % len(DRUGS) == 0

    treatment = {}
    for i, drug in enumerate(DRUGS):
        if scale == "linear":
            con = from_linear_scale(concentrations[i], max_dosage)
        elif scale == "log":
            con = from_log_scale(concentrations[i], max_dosage)
        elif scale == "real":
            con = concentrations[i]
        else:
            raise ValueError("Provided scale, is unknown.")

        treatment[drug] = float(con)

    return treatment



