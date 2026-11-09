"""Run from repository root: python habitat_nav.py --config config/habitat_nav.yaml."""
import argparse
import json
import math
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/habitat_nav.yaml')
    parser.add_argument('--environment-only', action='store_true', help='Exercise forward/left/right without loading NWM')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    # Keep optional heavy dependencies out of CLI help.
    from navigation.habitat_env import HabitatEnv
    env = HabitatEnv(cfg)
    records = []
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        rgb, goal = env.reset(cfg['start_pose'], cfg['goal_pose'])
        Image.fromarray(goal).save(output / 'goal.png')
        Image.fromarray(rgb).save(output / 'initial.png')
        if not args.environment_only:
            import torch
            torch.manual_seed(cfg['seed'])
            from navigation.nwm_policy import NWMPolicy
            policy = NWMPolicy(cfg, yaml.safe_load(Path(cfg['model_config']).read_text()),
                               yaml.safe_load(Path('config/data_config.yaml').read_text()))
            policy.reset(rgb, goal)
        success = False
        limit = 3 if args.environment_only else cfg['max_steps']
        for step in range(limit):
            distance, angle = env.errors()
            if not args.environment_only and distance <= cfg['success_distance'] and angle <= math.radians(cfg['success_angle_deg']):
                success = True
                break
            action, score = (step, None) if args.environment_only else policy.act()
            rgb, collided = env.step(action)
            distance, angle = env.errors()
            records.append(dict(step=step, action=action, predicted_cost=score, collided=collided,
                                distance=distance, yaw_error_rad=angle))
            Image.fromarray(rgb).save(output / f'{step:04d}.png')
            print(json.dumps(records[-1]), flush=True)
            if not args.environment_only:
                policy.observe(rgb)
                success = distance <= cfg['success_distance'] and angle <= math.radians(cfg['success_angle_deg'])
                if success:
                    break
        (output / 'result.json').write_text(json.dumps(dict(success=success,
            mode='environment-only' if args.environment_only else 'closed-loop',
            termination='environment_check' if args.environment_only else ('goal_reached' if success else 'step_limit'),
            config=cfg, steps=records), indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
