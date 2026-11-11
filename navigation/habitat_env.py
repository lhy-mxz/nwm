"""Small Habitat-Sim adapter; configuration follows sight-over-site's renderer.
Uses Habitat world coordinates directly (Y up, camera forward -Z).
"""
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis


class HabitatEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        sim = habitat_sim.SimulatorConfiguration()
        sim.scene_id = cfg['scene']
        sim.enable_physics = False
        sim.gpu_device_id = cfg['gpu']
        camera = habitat_sim.CameraSensorSpec()
        camera.uuid = 'rgb'
        camera.sensor_type = habitat_sim.SensorType.COLOR
        camera.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        camera.resolution = [240, 320]
        camera.hfov = cfg['hfov']
        camera.position = [0, cfg['camera_height'], 0]
        agent = habitat_sim.agent.AgentConfiguration()
        agent.radius = cfg['radius']
        agent.height = cfg['height']
        agent.sensor_specifications = [camera]
        agent.action_space = {
            0: habitat_sim.agent.ActionSpec('move_forward', habitat_sim.agent.ActuationSpec(amount=cfg['forward_m'])),
            1: habitat_sim.agent.ActionSpec('turn_left', habitat_sim.agent.ActuationSpec(amount=cfg['turn_deg'])),
            2: habitat_sim.agent.ActionSpec('turn_right', habitat_sim.agent.ActuationSpec(amount=cfg['turn_deg'])),
        }
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim, [agent]))
        self.sim.seed(cfg['seed'])
        if not self.sim.pathfinder.load_nav_mesh(cfg['navmesh']):
            self.close()
            raise ValueError('Cannot load NavMesh')

    def reset(self, start, goal):
        pf = self.sim.pathfinder
        start, goal = list(start), list(goal)
        if self.cfg.get('sight_over_site'):
            for pose in (start, goal):
                original = np.asarray(pose[:3], dtype=float)
                snapped = np.asarray(pf.snap_point(original))
                tolerance = self.cfg['sight_over_site'].get('snap_tolerance', .15)
                if not np.isfinite(snapped).all() or np.linalg.norm(snapped - original) > tolerance:
                    raise ValueError('Indexed pose cannot be snapped near floor_height_hint; check scene, NavMesh and coordinate settings')
                pose[:3] = snapped.tolist()
            self.cfg['start_pose'], self.cfg['goal_pose'] = start, goal
        for pose in (start, goal):
            if not pf.is_navigable(np.array(pose[:3])):
                raise ValueError('Start and goal must be navigable floor positions')
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = np.array(start[:3]), np.array(goal[:3])
        if not pf.find_path(path) or not 0 < path.geodesic_distance <= self.cfg['max_goal_distance']:
            raise ValueError(f'Goal is unreachable, identical, or beyond max_goal_distance={self.cfg["max_goal_distance"]}; measured geodesic={path.geodesic_distance}. Select another indexed episode or deliberately adjust the distance limit.')
        self.goal = goal
        self.set_pose(goal)
        target = self.observe().copy()
        self.set_pose(start)
        return self.observe(), target

    def set_pose(self, pose):
        self.sim.get_agent(0).set_state(habitat_sim.AgentState(
            position=np.array(pose[:3]), rotation=quat_from_angle_axis(np.deg2rad(pose[3]), np.array([0., 1., 0.]))))

    def observe(self):
        return self.sim.get_sensor_observations()['rgb'][..., :3].copy()

    def pose(self):
        from habitat_sim.utils.common import quat_rotate_vector
        state = self.sim.get_agent(0).get_state()
        forward = quat_rotate_vector(state.rotation, np.array([0., 0., -1.]))
        return np.asarray(state.position).copy(), float(np.arctan2(-forward[0], -forward[2]))

    def step(self, action):
        if isinstance(action, dict):
            return self.step_delta(action)
        result = self.sim.step(int(action))
        return result['rgb'][..., :3].copy(), bool(result.get('collided', False))

    def step_delta(self, action):
        """Kinematic local displacement with NavMesh filtering on every segment.

        Allows lateral motion; not a wheeled robot controller. Rotation is applied
        after translation, matching a relative-pose command in the pre-step frame.
        """
        from navigation.continuous import local_to_world
        if action.get('kind') != 'local_delta':
            raise ValueError('Unsupported continuous action')
        values = [action[k] for k in ('forward_m', 'left_m', 'yaw_rad')]
        if not np.isfinite(values).all():
            raise ValueError('Non-finite continuous action')
        position, yaw = self.pose()
        delta = local_to_world(values[0], values[1], yaw)
        segment_m = float(self.cfg.get('motion_segment_m', .05))
        if segment_m <= 0:
            raise ValueError('motion_segment_m must be positive')
        count = max(1, int(np.ceil(np.linalg.norm(delta) / segment_m)))
        collided = False
        for _ in range(count):
            requested = position + delta / count
            permitted = np.asarray(self.sim.pathfinder.try_step_no_sliding(position, requested))
            if not np.isfinite(permitted).all():
                raise RuntimeError('NavMesh returned a non-finite position')
            blocked = np.linalg.norm((permitted - requested)[[0, 2]]) > 1e-4
            position = permitted
            if blocked:
                collided = True
                break
        self.set_pose(position.tolist() + [float(np.rad2deg(yaw + values[2]))])
        return self.observe(), collided

    def errors(self):
        state = self.sim.get_agent(0).get_state()
        from habitat_sim.utils.common import quat_rotate_vector
        direction = quat_rotate_vector(state.rotation, np.array([0., 0., -1.]))
        yaw = np.arctan2(-direction[0], -direction[2])
        delta = yaw - np.deg2rad(self.goal[3])
        return float(np.linalg.norm(state.position - np.array(self.goal[:3]))), float(abs(np.arctan2(np.sin(delta), np.cos(delta))))

    def close(self):
        self.sim.close()
