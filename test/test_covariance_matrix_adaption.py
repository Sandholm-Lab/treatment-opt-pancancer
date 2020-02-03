import unittest
import os,sys,inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.search.evaluator import Evaluator
from src.search.covariance_matrix_adaption import cma_es
from src.reference_simulator.simulator import Simulator
from src.util.domain import UnitSimplex, Cube
from src.util.prepare_dict import prepare_dict
from src.env.objectives import Objective

EPS = 10e-8
MAX_ITER = 10

class TestObjective(Objective):
    # We use a replicator object because functions are not pickable
    def eval(self, rel_proliferations, action_dict):
        return np.average(rel_proliferations)

TEST_CONFIG = {
    "n_steps": 1,
    "cell_lines": ['DV90', 'HS695T', 'NCIH1092', 'PK59'],
    "objective": TestObjective(),
    "max_dosage": 8000,
    "domain": UnitSimplex(7),
    "scale": "linear"
}

class TestCovarianceMatrixAdaption(unittest.TestCase):
 
    def setUp(self):
        self.n_envs = 4
        self.evaluator = Evaluator(TEST_CONFIG, self.n_envs, store=True)
        self.domain = UnitSimplex(7)

    def test_cma_es(self):
        mu, obj, prolif = cma_es(self.evaluator, self.domain, MAX_ITER, verbose=True, seed=23)
        self.assertTrue(np.abs(obj - np.average(prolif)) < EPS)
        self.assertTrue(self.domain.contains(mu))

        # compare objective with sequential computation
        treatment = prepare_dict(mu.flatten(), max_dosage=TEST_CONFIG["max_dosage"], scale=TEST_CONFIG["scale"])
        print(treatment)
        prolifs = []
        for line in TEST_CONFIG["cell_lines"]:
            simulator = Simulator()
            simulator.initialize(line)
            prolifs.append(simulator.apply_treatment(treatment))
        o = TEST_CONFIG["objective"].eval(prolifs, treatment)
        self.assertTrue(np.abs(obj - o) <= EPS)

    def tearDown(self):
        # performs internal check if all environments terminate
        self.evaluator.terminate()

if __name__ == '__main__':
    unittest.main()
