"""Test NavMesh execution dispatch without requiring Habitat installation."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


def adapter_class():
    habitat = ModuleType('habitat_sim')
    common = ModuleType('habitat_sim.utils.common')
    common.quat_from_angle_axis = lambda *args: None
    modules = {'habitat_sim': habitat, 'habitat_sim.utils': ModuleType('habitat_sim.utils'),
               'habitat_sim.utils.common': common}
    spec = importlib.util.spec_from_file_location('tested_adapter', Path(__file__).parents[1] / 'navigation/habitat_env.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', modules):
        spec.loader.exec_module(module)
    return module.HabitatEnv


class MotionTests(unittest.TestCase):
    def execute(self, wall=False):
        cls = adapter_class()
        env = cls.__new__(cls)
        env.cfg = {'motion_segment_m': .05}
        calls = []
        def filtered(start, end):
            calls.append((start.copy(), end.copy()))
            return start if wall and end[2] < -.11 else end
        env.sim = SimpleNamespace(pathfinder=SimpleNamespace(try_step_no_sliding=filtered))
        env.pose = lambda: (np.zeros(3), 0.)
        saved = []
        env.set_pose = lambda pose: saved.append(pose)
        env.observe = lambda: 'image'
        image, collided = env.step(dict(kind='local_delta', forward_m=.25, left_m=0., yaw_rad=.2))
        self.assertEqual(image, 'image')
        self.assertAlmostEqual(saved[0][3], np.rad2deg(.2))
        self.assertTrue(all(np.linalg.norm(b-a) <= .050001 for a, b in calls))
        return saved[0], collided

    def test_free_motion(self):
        pose, collided = self.execute()
        self.assertFalse(collided)
        np.testing.assert_allclose(pose[:3], [0, 0, -.25])

    def test_blocked_motion_stops_at_wall(self):
        pose, collided = self.execute(wall=True)
        self.assertTrue(collided)
        np.testing.assert_allclose(pose[:3], [0, 0, -.1])
