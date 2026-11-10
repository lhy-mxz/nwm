"""Run from repository root: python habitat_nav.py --config config/habitat_nav.yaml."""
import argparse
import json
import math
from pathlib import Path
import yaml


def load_policy(cfg):
    import torch
    torch.manual_seed(cfg['seed'])
    from navigation.nwm_policy import NWMPolicy
    return NWMPolicy(cfg, yaml.safe_load(Path(cfg['model_config']).read_text()),
                     yaml.safe_load(Path('config/data_config.yaml').read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/habitat_nav.yaml')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--environment-only', action='store_true', help='Exercise forward/left/right without loading NWM')
    modes.add_argument('--model-only', action='store_true', help='Load NWM, VAE and LPIPS without starting Habitat')
    modes.add_argument('--single-step', choices=['forward', 'left', 'right'], help='Compare one NWM prediction with one real Habitat action; no CEM')
    parser.add_argument('--output-dir', help='Override result directory')
    parser.add_argument('--diffusion-steps', type=int, help='Override diffusion sampling steps')
    parser.add_argument('--list-episodes', action='store_true', help='List indexed sight-over-site tasks without loading Habitat or NWM')
    parser.add_argument('--resolve-episode', action='store_true', help='Print resolved scene paths and poses without starting simulation')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.diffusion_steps is not None:
        if not 1 <= args.diffusion_steps <= 1000:
            parser.error('--diffusion-steps must be between 1 and 1000')
        cfg['diffusion_steps'] = args.diffusion_steps
    if args.model_only:
        load_policy(cfg)
        print('PASS: NWM, VAE and LPIPS loaded; inference not exercised', flush=True)
        return
    from navigation.episodes import load_episode, list_episodes
    if args.list_episodes:
        list_episodes(cfg['sight_over_site'])
        return
    cfg = load_episode(cfg)
    if args.resolve_episode:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
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
            policy = load_policy(cfg)
            policy.reset(rgb, goal)
        if args.single_step:
            from navigation.probe import run_probe
            run_probe(env, policy, rgb, args.single_step, cfg, output)
            return
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
