# NWM Habitat closed-loop experiment

Run all commands from the NWM repository root. This is a short-range image-goal experiment, not a reproduction of the paper's navigation metrics.

## Requirements
Use your existing GPU Habitat-Sim environment, with NWM's torch/torchvision, diffusers, timm, LPIPS and other imports installed. Habitat-Lab and Stable Baselines are not required. Compatible package versions must be verified in that environment; the development cloud has not run Habitat or pretrained CUDA inference.

Provide a scene GLB, a compatible NavMesh (built for the configured agent radius/height), NWM EMA checkpoint, and the VAE and LPIPS weights. The last two may download on first use. Only load trusted model checkpoints.

Edit `config/habitat_nav.yaml`. Both poses use Habitat world coordinates: Y up, zero yaw faces -Z, positive yaw turns left. Positions are navigable floor positions; camera height is an additional sensor offset. Example poses are placeholders. Start and goal must be connected with geodesic distance within the configured limit.

```bash
python habitat_nav.py --config config/habitat_nav.yaml --environment-only
python habitat_nav.py --config config/habitat_nav.yaml
```

The environment-only mode executes forward, left, right and records observations without loading NWM. Use separate output directories for separate runs; existing named images/results are overwritten.

## Components
- `habitat_env.py`: scene/RGB/NavMesh setup, goal image capture, reset, native Habitat actions, pose error and cleanup. The configuration structure is informed by sight-over-site's HabitatRenderer; no runtime dependency on that repository.
- `planning.py`: categorical CEM searches executable forward/left/right sequences, with elite distribution updates and exploration smoothing.
- `nwm_policy.py`: loads the existing NWM model and inference wrapper, scores model rollouts using LPIPS against the goal RGB, and returns the first action. Actual observations replace the model's imagined history after execution.
- `../habitat_nav.py`: runs the episode, captures images and writes action/collision/error records to `result.json`.

## Experimental assumptions
Initial history repeats the start RGB four times, matching the notebook's bootstrap approach. Subsequent history contains actual simulator observations. RGB uses a 4:3 sensor and the repository's crop/resize/normalization. The camera settings are starting values, not demonstrated training-camera calibration.

Each primitive becomes a local NWM condition: forward maps to positive local x, left to positive yaw. Translation is scaled using `metric_waypoint_spacing`, then the original action min/max normalization. Even a zero translation must be normalized. Choose and calibrate `action_dataset` and `frames_per_action` against actual one-step rendered transitions before interpreting navigation results. Habitat and NWM have different visual/motion distributions.

This planner changes the original continuous CEM into categorical CEM so that imagined primitives are executable. It uses stochastic NWM samples, 50 diffusion steps by default, and terminal-image LPIPS only. Scores are noisy and there is no learned collision cost or guarantee of global navigation. Habitat performs collision filtering during actual movement. The environment reports ground-truth distance/yaw for evaluation only; the policy sees RGB and target RGB, not target coordinates. Success is environment-provided pose-threshold termination, not a learned STOP action. No SPL claim is made.

Validate environment motion/collisions, then model-versus-rendered one-step predictions, before assessing full navigation. No pretrained or Habitat end-to-end validation has been performed in the CPU-only cloud workspace.

## Existing sight-over-site indexed episodes

Use `config/habitat_nav_sos.yaml` and set only `sight_over_site.data_dir` (directory of scene `.npz` files) and `sight_over_site.scenes_dir` (Gibson scene directory). Existing server data can be referenced in place. No datasets are bundled in this repository.

```bash
python habitat_nav.py --config config/habitat_nav_sos.yaml --list-episodes
python habitat_nav.py --config config/habitat_nav_sos.yaml --resolve-episode
```

These commands do not start simulation or load models. Select zero-based `scene_index` (sorted NPZ filenames, as in the original evaluator) and `episode_index` (original `combos` index). The loader derives GLB/NavMesh paths from `scene_name`, and converts row/column positions using the original grid width, origin and 0.05 m cell size. It converts the stored start angle from radians with the original pi/2 orientation offset. Full manual poses remain supported by the original config.

The task uses the original `combos` target position as an **image-goal camera position**, not the original inspection endpoint. Goal angle in these files is a placeholder: this adapter defaults to the start camera's heading, with optional `goal_yaw_deg` override. This changes task semantics from inspection visibility to pose arrival. It does not reuse the inspection success labels or SPL reference lengths.

The floor-height hint defaults to zero, matching source dataset generation; the adapter snaps positions to NavMesh only within `snap_tolerance` and checks connectivity and geodesic distance. List output gives straight-line distances only. Some original tasks exceed the default 3 m range; choose an appropriate episode or explicitly adjust the limit, rather than silently changing the selected task. Verify resolved paths first, then run the environment-only command when ready. Neither scene files nor actual dataset compatibility were validated in the cloud because the data are absent.
