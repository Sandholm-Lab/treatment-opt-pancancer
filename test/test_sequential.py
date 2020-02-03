import unittest
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from src.reference_simulator.simulator import Simulator
from src.search.evaluator import Evaluator
from src.search.covariance_matrix_adaption import cma_es
from src.search.evaluate_search import best_sequential_search_result
from src.util.domain import SequentialSimplex, SequentialCube
from src.env.objectives import MultiWorstLinear
from src.env.drugs import empty_treatment
from src.util.prepare_dict import prepare_dict
import numpy as np

DIM = 7
N_STEPS = 3
LAMBD = 1e-4
MAX_ITER = 5
EPS = 10e-6

# TODO: Assert das die mehrstufige Ausführung zu monotoner Senkung

TEST_CONFIG = {
    "n_steps": N_STEPS,
    "cell_lines": ['DV90', 'SKMEL24'],
    "objective": MultiWorstLinear(LAMBD),
    "max_dosage": 8000,
    "domain": SequentialSimplex(DIM, N_STEPS),
    "scale": "linear"
}

class TestSequential(unittest.TestCase):
    """
    A test script to verify multiple function related to sequential experiments.
    """

    def setUp(self):
        self.lambd = 10 ** (-4.5)
        self.dim = DIM
        self.n_steps = N_STEPS
        self.evaluator = Evaluator(TEST_CONFIG, 10, store=False)
        self.domain = SequentialSimplex(self.dim, self.n_steps)
        self.sigma = np.eye(self.dim * self.n_steps) / self.dim
        self.objective = MultiWorstLinear(LAMBD)
        

    def test_sequential_simplex(self):
        seq_simplex = SequentialSimplex(self.dim, self.n_steps)
        vertex = np.zeros(self.dim * self.n_steps)
        for i in range(self.n_steps):
            vertex[i * self.dim] = 1
        self.assertTrue(seq_simplex.contains(np.zeros(self.dim * self.n_steps)))
        self.assertTrue(seq_simplex.contains(vertex))
        self.assertFalse(seq_simplex.contains(np.ones(self.dim * self.n_steps)))
        self.assertTrue(seq_simplex.contains(seq_simplex.center()))
        self.assertTrue(seq_simplex.contains(seq_simplex.uniform()))
        self.assertTrue(seq_simplex.contains(seq_simplex.normal(seq_simplex.center(), self.sigma)))

    def test_sequential_cube(self):
        seq_cube = SequentialCube(self.dim, self.n_steps)
        self.assertTrue(seq_cube.contains(np.zeros(self.dim * self.n_steps)))
        self.assertTrue(seq_cube.contains(np.ones(self.dim * self.n_steps)))
        self.assertTrue(seq_cube.contains(seq_cube.center()))
        self.assertTrue(seq_cube.contains(seq_cube.uniform()))
        self.assertTrue(seq_cube.contains(seq_cube.normal(seq_cube.center(), self.sigma)))

    def test_sequential_single_drug(self):
        # get single step result
        single_step_treatment = np.zeros(self.dim)
        single_step_treatment[0] = 1
        single_treat = prepare_dict(single_step_treatment, max_dosage=8000)
        simulator = Simulator()
        simulator.initialize("SKMEL24")
        sp1 = simulator.apply_treatment(single_treat)
        sp2 = simulator.apply_treatment(single_treat)
        sp3 = simulator.apply_treatment(single_treat)
        self.assertAlmostEqual(sp2, sp1 * sp1)
        self.assertAlmostEqual(sp3, sp1 ** 3)
        self.assertAlmostEqual(sp3, sp1 * sp2)

        # get multi-step result
        multi_step_treatment = np.zeros(self.dim * self.n_steps)
        for i in range(self.n_steps):
            multi_step_treatment[i * self.dim] = 1
        ys, prolifs = self.evaluator.evaluate([multi_step_treatment])

        self.assertAlmostEqual(sp1 ** 3, prolifs[0][1])
        self.assertAlmostEqual(sp1 ** 3 + LAMBD * 3 * 8000, ys[0])

    def test_sequential_evaluation(self):
        seq_simplex = SequentialSimplex(self.dim, self.n_steps)
        x = seq_simplex.uniform()
        treats = [prepare_dict(x[i * self.dim:(i + 1) * self.dim], max_dosage=TEST_CONFIG["max_dosage"], scale=TEST_CONFIG["scale"]) for i in range(self.n_steps)]

        # use evaluator
        ys, prolifs = self.evaluator.evaluate([x])
        # compare with SKMEL24 evaluation of single steps
        p = 1
        for i in range(self.n_steps):
            simulator = Simulator()
            simulator.initialize("SKMEL24")
            p *= simulator.apply_treatment(treats[i])
        self.assertAlmostEqual(prolifs[0][1], p)

    def test_sequential_cma_es(self):
        mu, obj, prolif = cma_es(self.evaluator, self.domain, MAX_ITER, verbose=True, seed=23)
        # generate total dosage
        total_dosage = sum([8000 * x for x in mu])
        self.assertTrue(np.abs(obj - (np.max(prolif) + LAMBD * total_dosage)) < EPS)
        self.assertTrue(self.domain.contains(mu))

        # compare objective with sequential computation
        mu = mu.flatten()
        treats = [prepare_dict(mu[i * self.dim:(i + 1) * self.dim], max_dosage=TEST_CONFIG["max_dosage"], scale=TEST_CONFIG["scale"]) for i in range(self.n_steps)]
        cum_treat = empty_treatment()
        prolifs = []
        for line in TEST_CONFIG["cell_lines"]:
            simulator = Simulator()
            simulator.initialize(line)
            for t in treats:
                p = simulator.apply_treatment(t)
            prolifs.append(p)
        for t in treats:
            for k in t:
                cum_treat[k] += t[k]
        o = TEST_CONFIG["objective"].eval(prolifs, cum_treat)
        self.assertTrue(np.abs(obj - o) <= EPS)

    def test_best_sequential_search_result(self):
        treatments, concentration, rel_prolif, objective = best_sequential_search_result("intestine", "./artifacts/sequential/", "2step_worst_simplex_-4_5", 2, lambd=self.lambd, obj="worst", max_dosage=8000, verification=True)
        total_concentration = 0
        for t in treatments:
            for k in t:
                total_concentration += t[k]
        self.assertAlmostEqual(total_concentration, concentration)
        ref_obj = np.max(rel_prolif) + self.lambd * concentration 
        self.assertAlmostEqual(objective, ref_obj)

    def best_interpolated_multi_search_result(self):
        treatments, concentration, rel_prolif, objective = best_sequential_search_result("intestine", "./artifacts/multi/", "worst_simplex_-4_5", 2, lambd=self.lambd, obj="worst", max_dosage=8000, verification=True)
        total_concentration = 0
        for t in treatments:
            for k in t:
                total_concentration += t[k]
        self.assertAlmostEqual(total_concentration, concentration)
        ref_obj = np.max(rel_prolif) + self.lambd * concentration 
        self.assertAlmostEqual(objective, ref_obj)

    def tearDown(self):
        # performs internal check if all environments terminate
        self.evaluator.terminate()

if __name__ == '__main__':
    unittest.main()
