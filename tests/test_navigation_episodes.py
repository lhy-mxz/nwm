import tempfile
from pathlib import Path
import unittest
import numpy as np
from navigation.episodes import load_episode


class EpisodesTests(unittest.TestCase):
    def test_source_grid_conversion_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'room').mkdir()
            for suffix in ('glb', 'navmesh'):
                (root / 'room' / f'room.{suffix}').touch()
            np.savez(root / 'room.npz', scene_name='room', origin=[10., 20.],
                     occupancy_grid=np.zeros((80, 100)),
                     combos=np.array([[[2., 4., 0.], [6., 8., 0.]],
                                      [[10., 20., np.pi/2], [12., 22., 0.]]]))
            cfg = {'sight_over_site': {'data_dir': directory, 'scenes_dir': directory,
                                      'scene_index': 0, 'episode_index': 1}}
            result = load_episode(cfg)
            np.testing.assert_allclose(result['start_pose'], [14., 0., 20.5, 180.])
            np.testing.assert_allclose(result['goal_pose'], [13.9, 0., 20.6, 180.])
            self.assertEqual(result['episode_source']['episode_index'], 1)
            cfg['sight_over_site']['episode_index'] = 2
            with self.assertRaises(ValueError):
                load_episode(cfg)

    def test_manual_mode(self):
        cfg = {'start_pose': [0, 0, 0, 0]}
        self.assertEqual(load_episode(cfg), cfg)


if __name__ == '__main__':
    unittest.main()
