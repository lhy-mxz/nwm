"""CPU artifact/dispatch check; does not validate pretrained model or Habitat."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires PyTorch')
class ProbeTests(unittest.TestCase):
    def test_fixed_action_and_outputs(self):
        from navigation.probe import run_probe
        position = np.zeros(3)
        actions = []
        def step(action):
            actions.append(action)
            position[2] -= .25
            return np.zeros((240, 320, 3), dtype=np.uint8), False
        env = SimpleNamespace(sim=SimpleNamespace(get_agent=lambda _: SimpleNamespace(
            get_state=lambda: SimpleNamespace(position=position))), step=step)
        predicted_actions = []
        def predict(history, action):
            predicted_actions.extend(action.tolist())
            return torch.full((1, 3, 224, 224), -1.)
        policy = SimpleNamespace(device='cpu', history=torch.full((4, 3, 224, 224), -1.),
            predict=predict, metric=lambda a, b: (a - b).square().mean())
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.synchronize'):
            output = Path(directory)
            run_probe(env, policy, None, 'forward', {'diffusion_steps': 10}, output)
            report = json.loads((output / 'probe.json').read_text())
            self.assertEqual(actions, [0])
            self.assertEqual(predicted_actions, [0])
            self.assertAlmostEqual(report['actual_translation_m'], .25)
            self.assertEqual(report['predicted_actual_lpips'], 0.)
            from PIL import Image
            with Image.open(output / 'comparison.png') as image:
                self.assertEqual(image.size, (672, 248))
