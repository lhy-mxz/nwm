import unittest
import numpy as np
from navigation.continuous import make_deltas, gaussian_search, local_to_world


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.low = np.array([-2.5, -4.])
        self.high = np.array([5., 4.])

    def test_heading_and_final_perturbation(self):
        # Physical XY = (1, 1): first yaw is pi/4; last adds pi * parameter.
        xy = 2 * (np.array([1., 1.]) - self.low)/(self.high-self.low)-1
        result = make_deltas(np.array([[*xy, .1]]), 3, self.low, self.high)
        np.testing.assert_allclose(result[0, :, 2], [np.pi/4, 0, .1*np.pi])
        np.testing.assert_allclose(result[0, :, :2], np.tile(xy, (3, 1)))

    def test_single_round_and_final_mean(self):
        calls = []
        def score(a):
            calls.append(a.copy())
            return np.arange(len(a), dtype=float)
        final, cost, trace = gaussian_search(score, np.random.default_rng(0),
            [-.1, 0, 0], [.02, .1, .1], self.low, self.high, 8, 6, 2)
        self.assertEqual([len(a) for a in calls], [6, 1])
        expected_mu = calls[0][:2, -1].mean(0)
        np.testing.assert_allclose(trace['updated_mu'], expected_mu)
        np.testing.assert_allclose(final, make_deltas(expected_mu[None], 8, self.low, self.high)[0])

    def test_world_axes(self):
        np.testing.assert_allclose(local_to_world(1, 0, 0), [0, 0, -1])
        np.testing.assert_allclose(local_to_world(0, 1, 0), [-1, 0, 0])
        np.testing.assert_allclose(local_to_world(1, 0, np.pi/2), [-1, 0, 0], atol=1e-12)

    def test_bad_elite_count(self):
        with self.assertRaises(ValueError):
            gaussian_search(None, None, [0]*3, [1]*3, self.low, self.high, elites=1)
