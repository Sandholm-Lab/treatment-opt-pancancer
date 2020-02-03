import unittest
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.baseline.evaluate import build_combined_single_frame, build_combined_dual_frame, best_single_treatment, best_dual_treatment, best_sequential_single_treatment, best_sequential_dual_treatment
from src.baseline.evaluate import best_single_treatment_by_dosage, best_dual_treatment_by_dosage
from src.env.cell_lines import retrieve_lines
from src.env.drugs import DRUGS
import numpy as np
import pandas as pd

EPS = 10e-8
PATH = "./artifacts/baselines/"
RATIOS = [x * 5 for x in range(21)] # 5% steps
VERIFICATION = False

class TestEvaluate(unittest.TestCase):
 
    def setUp(self):
        self.tissue = "skin"
        self.cell_lines = retrieve_lines(self.tissue)
        self.lambd = 10 ** -4.5
        self.rows = [10, 47, 430, 1150]
        self.n_steps = 2

    def test_build_combined_single_frame(self):
        combined = build_combined_single_frame(self.cell_lines)
        # manually check a few entries
        for i, line in enumerate(self.cell_lines):
            data = pd.read_pickle(PATH + line + "_baseline.pkl")
            self.assertTrue(np.allclose(combined["concentration"].values, data["concentration"].values))
            for d in DRUGS:
                for r in self.rows:
                    self.assertTrue(np.abs(combined[d][r][i] - data[d][r]) < EPS)

    def test_build_combined_dual_frame(self):
        combined = build_combined_dual_frame(self.cell_lines)
        # manually check a few entries
        for i, line in enumerate(self.cell_lines):
            data = pd.read_pickle(PATH + line + "_dual.pkl")
            self.assertTrue(np.allclose(combined["concentration"].values, data["concentration"].values))
            for ratio in RATIOS:
                for r in self.rows:
                    self.assertTrue(np.abs(combined[ratio][r][i] - data[ratio][r]) < EPS)

    def test_best_single_treatment(self):
        comb_data = build_combined_single_frame(self.cell_lines)
        _, concentration, prolif, obj = best_single_treatment(self.cell_lines, lambd=0, obj="avg", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_single_treatment(self.cell_lines, lambd=0, obj="avg", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.average(prolif))
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_single_treatment(self.cell_lines, lambd=self.lambd, obj="avg", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj =  best_single_treatment(self.cell_lines, lambd=self.lambd, obj="avg", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.average(prolif) + self.lambd * concentration)
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_single_treatment(self.cell_lines, lambd=0, obj="worst", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_single_treatment(self.cell_lines, lambd=0, obj="worst", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.max(prolif))
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_single_treatment(self.cell_lines, lambd=self.lambd, obj="worst", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_single_treatment(self.cell_lines, lambd=self.lambd, obj="worst", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.max(prolif) + self.lambd * concentration)
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)

    def test_best_sequential_single_treatment(self):
        comb_data = build_combined_single_frame(self.cell_lines)
        _, concentration, prolif, obj = best_sequential_single_treatment(self.cell_lines, self.n_steps, lambd=0, obj="avg", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_sequential_single_treatment(self.cell_lines, self.n_steps, lambd=0, obj="avg", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.average(prolif))
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_sequential_single_treatment(self.cell_lines, self.n_steps, lambd=self.lambd, obj="avg", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_sequential_single_treatment(self.cell_lines, self.n_steps, lambd=self.lambd, obj="avg", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.average(prolif) + self.lambd * concentration)
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_sequential_single_treatment(self.cell_lines, self.n_steps, lambd=0, obj="worst", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_sequential_single_treatment(self.cell_lines, self.n_steps, lambd=0, obj="worst", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.max(prolif))
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_sequential_single_treatment(self.cell_lines, self.n_steps, lambd=self.lambd, obj="worst", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_sequential_single_treatment(self.cell_lines, self.n_steps, lambd=self.lambd, obj="worst", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.max(prolif) + self.lambd * concentration)
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)

    def test_best_dual_treatment(self):
        comb_data = build_combined_dual_frame(self.cell_lines)
        _, concentration, prolif, obj = best_dual_treatment(self.cell_lines, lambd=0, obj="avg", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_dual_treatment(self.cell_lines, lambd=0, obj="avg", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.average(prolif))
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_dual_treatment(self.cell_lines, lambd=self.lambd, obj="avg", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_dual_treatment(self.cell_lines, lambd=self.lambd, obj="avg", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.average(prolif) + self.lambd * concentration)
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_dual_treatment(self.cell_lines, lambd=0, obj="worst", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_dual_treatment(self.cell_lines, lambd=0, obj="worst", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.max(prolif))
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_dual_treatment(self.cell_lines, lambd=self.lambd, obj="worst", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_dual_treatment(self.cell_lines, lambd=self.lambd, obj="worst", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.max(prolif) + self.lambd * concentration)
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)

    def test_best_sequential_dual_treatment(self):
        comb_data = build_combined_dual_frame(self.cell_lines)
        _, concentration, prolif, obj = best_sequential_dual_treatment(self.cell_lines, self.n_steps, lambd=0, obj="avg", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_sequential_dual_treatment(self.cell_lines, self.n_steps, lambd=0, obj="avg", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.average(prolif))
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_sequential_dual_treatment(self.cell_lines, self.n_steps, lambd=self.lambd, obj="avg", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_sequential_dual_treatment(self.cell_lines, self.n_steps, lambd=self.lambd, obj="avg", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.average(prolif) + self.lambd * concentration)
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_sequential_dual_treatment(self.cell_lines, self.n_steps, lambd=0, obj="worst", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_sequential_dual_treatment(self.cell_lines, self.n_steps, lambd=0, obj="worst", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.max(prolif))
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)
        _, concentration, prolif, obj = best_sequential_dual_treatment(self.cell_lines, self.n_steps, lambd=self.lambd, obj="worst", verification=VERIFICATION)
        _, comb_concentration, comb_prolif, comb_obj = best_sequential_dual_treatment(self.cell_lines, self.n_steps, lambd=self.lambd, obj="worst", verification=False, comb_data=comb_data)
        self.assertAlmostEqual(obj, np.max(prolif) + self.lambd * concentration)
        self.assertAlmostEqual(prolif, comb_prolif)
        self.assertAlmostEqual(obj, comb_obj)
        self.assertAlmostEqual(concentration, comb_concentration)

    def test_best_single_treatment_by_dosage(self):
        best_drugs, concentrations, best_prolifs = best_single_treatment_by_dosage(self.cell_lines, obj="worst", drugs=DRUGS, path="./artifacts/baselines/")
        combined = build_combined_single_frame(self.cell_lines)
        self.assertTrue(len(best_drugs) == len(concentrations) and len(best_drugs) == len(best_prolifs))
        for d in DRUGS:
            for r in self.rows:
                self.assertAlmostEqual(concentrations[r], combined["concentration"][r]) # NOTE: This assumes 1nM steps in baseline
                self.assertTrue(np.max(combined[best_drugs[r]][r]) <= best_prolifs[r])
                self.assertTrue(best_prolifs[r] <= np.max(combined[d][r]))

    def test_best_dual_treatment_by_dosage(self):
        best_ratios, concentrations, best_prolifs = best_dual_treatment_by_dosage(self.cell_lines, obj="worst", ratios=RATIOS, path="./artifacts/baselines/")
        combined = build_combined_dual_frame(self.cell_lines)
        self.assertTrue(len(best_ratios) == len(concentrations) and len(best_ratios) == len(best_prolifs))
        for ratio in RATIOS:
            for r in self.rows:
                self.assertAlmostEqual(concentrations[r], combined["concentration"][r]) # NOTE: This assumes 1nM steps in baseline
                self.assertTrue(np.max(combined[best_ratios[r]][r]) <= best_prolifs[r])
                self.assertTrue(best_prolifs[r] <= np.max(combined[ratio][r]))

if __name__ == '__main__':
    unittest.main()
