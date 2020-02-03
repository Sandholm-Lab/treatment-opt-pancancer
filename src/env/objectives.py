from abc import ABC, abstractmethod
import numpy as np

class Objective(ABC):
    @abstractmethod
    def eval(self, rel_proliferations, action_dict):
        raise NotImplementedError()

class SingleLinear(Objective):
    def __init__(self, lambd):
        self.lambd = lambd

    def eval(self, rel_proliferations, action_dict):
        assert len(rel_proliferations) == 1, "Objective not designed for multi-cell experiments."
        total_dosage = 0
        for k in action_dict:
            assert action_dict[k] >= 0, "Negative dosage is not allowed."
            total_dosage += action_dict[k]
        val = rel_proliferations[0] + self.lambd * total_dosage 
        return val

class MultiAvgLinear(Objective):
    def __init__(self, lambd):
        self.lambd = lambd

    def eval(self, rel_proliferations, action_dict):
        total_dosage = 0
        for k in action_dict:
            assert action_dict[k] >= 0, "Negative dosage is not allowed."
            total_dosage += action_dict[k]
        val = np.average(rel_proliferations) + self.lambd * total_dosage 
        return val

class MultiWorstLinear(Objective):
    def __init__(self, lambd):
        self.lambd = lambd
    
    def eval(self, rel_proliferations, action_dict):
        total_dosage = 0
        for k in action_dict:
            assert action_dict[k] >= 0, "Negative dosage is not allowed."
            total_dosage += action_dict[k]
        val = max(rel_proliferations) + self.lambd * total_dosage 
        return val

# -------------------------------------------------------------------------------------------------

def retrieve_multi_objective(obj, lambd):
    if obj == "avg":
        objective = MultiAvgLinear(lambd)
    elif obj == "worst":
        objective = MultiWorstLinear(lambd)
    else:
        raise ValueError("The specified objective type is unknown.")
    return objective
