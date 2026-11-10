"""Read sight-over-site scene metadata without importing Habitat or NWM."""
from pathlib import Path
import numpy as np


def load_episode(cfg):
    settings = cfg.get('sight_over_site')
    if not settings:
        return dict(cfg)
    root = Path(settings['data_dir']).expanduser()
    files = sorted(root.glob('*.npz'))
    index = int(settings.get('scene_index', 0))
    if not 0 <= index < len(files):
        raise ValueError(f'No scene at index {index}; found {len(files)} .npz files in {root}')
    with np.load(files[index], allow_pickle=False) as data:
        name = str(data['scene_name'].item())
        combos = data['combos']
        combo = int(settings.get('episode_index', 0))
        if not 0 <= combo < len(combos):
            raise ValueError(f'Episode index {combo} out of range [0, {len(combos)})')
        start, target = np.asarray(combos[combo], dtype=float)
        origin = np.asarray(data['origin'], dtype=float)
        width = data['occupancy_grid'].shape[1]
        scale = float(settings.get('meters_per_pixel', .05))
        def position(p):
            # Matches sight-over-site HabitatRenderer.grid_to_world X/Z.
            # Ground height is resolved against NavMesh at environment reset.
            return [float(origin[0] + (width - p[1]) * scale),
                    float(settings.get('floor_height_hint', 0.0)),
                    float(origin[1] + p[0] * scale)]
        start_pos, goal_pos = position(start), position(target)
        start_yaw = float(np.rad2deg(start[2] + np.pi / 2))
        # Dataset goal angle is a placeholder. Define image-goal view explicitly.
        goal_yaw = float(settings.get('goal_yaw_deg', start_yaw))
    scene_dir = Path(settings['scenes_dir']).expanduser() / name
    result = dict(cfg)
    result.update(scene=str(scene_dir / f'{name}.glb'),
                  navmesh=str(scene_dir / f'{name}.navmesh'),
                  start_pose=start_pos + [start_yaw], goal_pose=goal_pos + [goal_yaw],
                  episode_source=dict(file=str(files[index]), scene_name=name,
                                      scene_index=index, episode_index=combo))
    for key in ('scene', 'navmesh'):
        if not Path(result[key]).is_file():
            raise FileNotFoundError(f'{key}: {result[key]}')
    return result


def list_episodes(settings):
    root = Path(settings['data_dir']).expanduser()
    files = sorted(root.glob('*.npz'))
    if not files:
        raise ValueError(f'No .npz files in {root}')
    scale = float(settings.get('meters_per_pixel', .05))
    for i, path in enumerate(files):
        with np.load(path, allow_pickle=False) as data:
            print(f"scene_index={i}: {path.name}, scene={data['scene_name'].item()}, episodes={len(data['combos'])}")
            for j, pair in enumerate(data['combos']):
                distance = np.linalg.norm(pair[0, :2] - pair[1, :2]) * scale
                print(f'  episode_index={j}: straight-line distance={distance:.2f} m (not geodesic)')
