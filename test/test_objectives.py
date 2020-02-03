import unittest
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.env.objectives import SingleLinear, MultiAvgLinear, MultiWorstLinear
from src.env.drugs import empty_treatment


class TestObjectives(unittest.TestCase):
 
    def setUp(self):
        self.lambd = 1
        self.treatment = empty_treatment()
        for k in self.treatment:
            self.treatment[k] = 1

        self.p1 = 1
        self.p2 = 2
        self.p3 = 3

    def test_single_linear(self):
        obj = SingleLinear(0)
        self.assertAlmostEqual(self.p1, obj.eval([self.p1], self.treatment))
        obj = SingleLinear(self.lambd)
        self.assertAlmostEqual(self.p1 + self.lambd * 7, obj.eval([self.p1], self.treatment))

    def test_multi_avg_linear(self):
        obj = MultiAvgLinear(0)
        self.assertAlmostEqual(2, obj.eval([self.p1, self.p2, self.p3], self.treatment))
        obj = MultiAvgLinear(self.lambd)
        self.assertAlmostEqual(2 + self.lambd * 7, obj.eval([self.p1, self.p2, self.p3], self.treatment))

    def test_multi_worst_linear(self):
        obj = MultiWorstLinear(0)
        self.assertAlmostEqual(3, obj.eval([self.p1, self.p2, self.p3], self.treatment))
        obj = MultiWorstLinear(self.lambd)
        self.assertAlmostEqual(3 + self.lambd * 7, obj.eval([self.p1, self.p2, self.p3], self.treatment))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
