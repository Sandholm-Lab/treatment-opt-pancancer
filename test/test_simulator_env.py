import unittest
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.env.simulator_env import SimulatorEnv
from src.reference_simulator.simulator import Simulator
from src.util.domain import UnitSimplex
from util.prepare_dict import prepare_dict
import numpy as np

EPS = 10e-8

class TestObjective():
    # We use a replicator object because functions are not pickable
    def eval(self, rel_proliferations, action_dict):
        return rel_proliferations

class TestSimulatorEnv(unittest.TestCase):
 
    def setUp(self):
        self.n_steps = 1
        self.cell_lines = [
            'DV90',
            'HS695T',
            'NCIH1092',
            'PK59',
        ]
        self.max_dosage = 8000
        self.reward_function = lambda x : x
        self.penalty_function= lambda x: 0
        self.env = SimulatorEnv(self.n_steps, self.cell_lines, self.max_dosage, TestObjective(), UnitSimplex(7), "linear")
        self.treatment = np.array([0.35, 0.05, 0.1, 0.1, 0.1, 0.15, 0.05])

    def test_worker_reset(self):
        # This test performs two resets of the environment in a row

        prolifs = self.env.reset()
        self.assertTrue(np.allclose(prolifs, np.ones(len(self.cell_lines))))

        prolifs = self.env.reset()
        self.assertTrue(np.allclose(prolifs, np.ones(len(self.cell_lines))))

    def test_experiment(self):
        # This test performs two experiments with the environment in a row
        _ = self.env.reset()
        steps = 0

        done = False
        while not done:
            steps += 1
            _, reward, done, _ = self.env.step(self.treatment)
        self.assertEqual(steps, self.n_steps)

        # compare output with regular simulation
        treat = prepare_dict(self.treatment,  max_dosage=self.max_dosage)
        for i, line in enumerate(self.cell_lines):
            simulator = Simulator()
            simulator.initialize(line)
            r = simulator.apply_treatment(treat)
            self.assertTrue(np.abs(r - reward[i]) < EPS)

        _ = self.env.reset()
        steps = 0

        done = False
        while not done:
            steps += 1
            _, reward, done, _ = self.env.step(self.treatment)
        self.assertEqual(steps, self.n_steps)

        # compare output with regular simulation
        treat = prepare_dict(self.treatment,  max_dosage=self.max_dosage)
        for i, line in enumerate(self.cell_lines):
            simulator = Simulator()
            simulator.initialize(line)
            r = simulator.apply_treatment(treat)
            self.assertTrue(np.abs(r - reward[i]) < EPS)

    def tearDown(self):
        self.env.terminate()

if __name__ == '__main__':
    unittest.main()
