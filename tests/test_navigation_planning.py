import unittest
import numpy as np
from navigation.planning import search


class PlanningTests(unittest.TestCase):
    def test_known_optimum(self):
        target = np.array([0, 1, 2])
        actions, cost = search(lambda a: (a != target).sum(1), np.random.default_rng(0), 3, 64, 8, 4)
        np.testing.assert_array_equal(actions, target)
        self.assertEqual(cost, 0)

    def test_invalid_scores(self):
        with self.assertRaises(ValueError):
            search(lambda a: np.full(len(a), np.nan), np.random.default_rng(0))

    def test_invalid_elites(self):
        with self.assertRaises(ValueError):
            search(None, np.random.default_rng(0), candidates=2, elites=3)


if __name__ == '__main__':
    unittest.main()
