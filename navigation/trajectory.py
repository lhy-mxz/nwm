"""Actual executed X/Z trajectory (not a collision map)."""
from pathlib import Path
import numpy as np


def save_trajectory(records, cfg, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points = np.array([cfg['start_pose'][:3]] + [r['position_after'] for r in records])
    goal = np.array(cfg['goal_pose'][:3])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(points[:, 0], points[:, 2], '-o', markersize=3, label='Executed trajectory')
    ax.scatter(*points[0, [0, 2]], marker='s', label='Start')
    ax.scatter(*goal[[0, 2]], marker='*', s=130, label='Goal')
    for index, record in enumerate(records):
        p = points[index + 1]
        angle = record['yaw_after_rad']
        ax.arrow(p[0], p[2], -.1*np.sin(angle), -.1*np.cos(angle), width=.005, alpha=.5)
        if record['collided']:
            ax.scatter(p[0], p[2], c='red', marker='x')
    ax.set(xlabel='Habitat X (m)', ylabel='Habitat Z (m)', title='Executed path; arrows show heading')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / 'trajectory.png', dpi=140)
    plt.close(fig)
