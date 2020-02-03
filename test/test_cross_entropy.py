import unittest
import os,sys,inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.search.evaluator import Evaluator
from src.search.cross_entropy import cross_entropy_method
from src.util.domain import UnitSimplex
from src.env.objectives import Objective

EPS = 10e-8
MAX_ITER = 5

class TestObjective(Objective):
    # We use a replicator object because functions are not pickable
    def eval(self, rel_proliferations, action_dict):
        return np.average(rel_proliferations)

TEST_CONFIG = {
    "n_steps": 1,
    "cell_lines": ['DV90', 'HS695T'],
    "objective": TestObjective(),
    "max_dosage": 8000,
    "domain": UnitSimplex(7),
    "scale": "linear"
}

class TestCrossEntropy(unittest.TestCase):
 
    def setUp(self):
        self.n_envs = 4
        self.evaluator = Evaluator(TEST_CONFIG, self.n_envs, store=True)
        self.domain = UnitSimplex(7)

    def test_cross_entropy(self):
        mu, obj, prolif = cross_entropy_method(self.evaluator, self.domain, MAX_ITER, 20, 10, 7, verbose=True, seed=23)
        self.assertTrue(np.abs(obj - np.average(prolif)) < EPS)
        self.assertTrue(self.domain.contains(mu))

    def tearDown(self):
        # performs internal check if all environments terminate
        self.evaluator.terminate()

if __name__ == '__main__':
    unittest.main()
