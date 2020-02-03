import unittest
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.search.evaluator import Evaluator
from src.reference_simulator.simulator import Simulator
from src.util.domain import UnitSimplex
from util.prepare_dict import prepare_dict
import numpy as np

EPS = 10e-6
EVALS = 5

class TestObjective():
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

class TestEvaluator(unittest.TestCase):
 
    def setUp(self):
        self.n_envs = 2
        self.n_steps = 3
        self.evaluator = Evaluator(TEST_CONFIG, self.n_envs, store=True)
        self.xs = [np.random.uniform(0, 1, 7) for i in range(EVALS)]
        self.xs = [x / sum(x + EPS) for x in self.xs]

    def test_evaluate(self):
        ys, prolifs = self.evaluator.evaluate(self.xs)
        ys, prolifs = self.evaluator.evaluate(self.xs)

        # compare results with direct serial execution
        for i, x in enumerate(self.xs):
            self.assertTrue(np.abs(ys[i] - np.average(prolifs[i])) < EPS)
            avg = 0
            for j, line in enumerate(TEST_CONFIG["cell_lines"]):
                treat = prepare_dict(x, max_dosage=TEST_CONFIG["max_dosage"])
                simulator = Simulator()
                simulator.initialize(line)
                r = simulator.apply_treatment(treat)
                self.assertTrue(np.abs(prolifs[i][j] - r) < EPS)
                avg += r
            avg /= len(TEST_CONFIG["cell_lines"])
            self.assertTrue(np.abs(avg - ys[i]) < EPS)

    def test_buffer(self):
        # test if things get stored in buffer correctly
        _, _ = self.evaluator.evaluate(self.xs)
        buffer_dict = self.evaluator.get_res_dict()
        self.assertEqual(len(buffer_dict[TEST_CONFIG["cell_lines"][0]]["relative_proliferation"]), EVALS)
        
        # compare buffer content with direct serial execution
        for i, x in enumerate(self.xs):
            for line in TEST_CONFIG["cell_lines"]:
                treat = prepare_dict(x, max_dosage=TEST_CONFIG["max_dosage"])
                simulator = Simulator()
                simulator.initialize(line)
                prolif = simulator.apply_treatment(treat)
                self.assertTrue(np.abs(prolif - buffer_dict[line]["relative_proliferation"][i]) <= EPS)

    def test_repeated_evaluation(self):
        SEQUENTIAL_CONFIG = {
            "n_steps": 3,
            "cell_lines": ['DV90', 'HS695T'],
            "objective": TestObjective(),
            "max_dosage": 8000,
            "domain": UnitSimplex(7),
            "scale": "linear"
        }

        repeated_evaluator = Evaluator(SEQUENTIAL_CONFIG, self.n_envs, store=True, repeated=True)
        x = np.array([0.5, 0.5, 0, 0, 0, 0, 0])
        treats = [prepare_dict(x, max_dosage=TEST_CONFIG["max_dosage"], scale=TEST_CONFIG["scale"]) for _ in range(self.n_steps)]

        # use evaluator
        _, prolifs = repeated_evaluator.evaluate([x]) # write test to check y
        p = 1
        for i in range(self.n_steps):
            simulator = Simulator()
            simulator.initialize("HS695T")
            p *= simulator.apply_treatment(treats[i])
        print("p: ", p)
        print("prolis: ", prolifs)
        self.assertAlmostEqual(prolifs[0][1], p)
        repeated_evaluator.terminate()

    def test_evaluate_without_pd(self):
        SEQUENTIAL_CONFIG = {
            "n_steps": 2,
            "cell_lines": ['DV90', 'HS695T'],
            "objective": TestObjective(),
            "max_dosage": 8000,
            "domain": UnitSimplex(7),
            "scale": "linear"
        }

        repeated_evaluator = Evaluator(SEQUENTIAL_CONFIG, self.n_envs, store=True, repeated=True, allow_pd=False)
        x = np.array([0.0, 0.5, 0.5, 0, 0, 0, 0])
        treats = [prepare_dict(x, max_dosage=TEST_CONFIG["max_dosage"], scale=TEST_CONFIG["scale"]) for _ in range(self.n_steps)]

        # use evaluator
        _, prolifs = repeated_evaluator.evaluate([x[1:]]) # write test to check y
        p = 1
        for i in range(SEQUENTIAL_CONFIG["n_steps"]):
            simulator = Simulator()
            simulator.initialize("HS695T")
            p *= simulator.apply_treatment(treats[i])
        print("p: ", p)
        print("prolis: ", prolifs)
        self.assertAlmostEqual(prolifs[0][1], p)
        repeated_evaluator.terminate()

    def tearDown(self):
        # performs internal check if all environments terminate
        self.evaluator.terminate()

if __name__ == '__main__':
    unittest.main()
