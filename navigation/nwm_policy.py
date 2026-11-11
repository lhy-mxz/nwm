"""NWM image scoring with executable action conditions; no dataset or DDP."""
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
import lpips
from models import CDiT_models
from diffusion import create_diffusion
from isolated_nwm_infer import model_forward_wrapper
from misc import transform
from navigation.planning import search
from navigation.continuous import gaussian_search
from pathlib import Path
import yaml


class NWMPolicy:
    def __init__(self, cfg, model_cfg, data_cfg):
        if not torch.cuda.is_available():
            raise RuntimeError('NWM Habitat navigation requires CUDA')
        self.cfg, self.device = cfg, f"cuda:{cfg['gpu']}"
        self.context = model_cfg['context_size']
        self.latent = model_cfg['image_size'] // 8
        if model_cfg['image_size'] != 224:
            raise ValueError('Current image preprocessing requires image_size=224')
        print(f"Loading NWM checkpoint: {cfg['checkpoint']}", flush=True)
        model = CDiT_models[model_cfg['model']](input_size=self.latent, context_size=self.context)
        checkpoint = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['ema'], strict=True)
        del checkpoint
        model.eval().to(self.device)
        print(f"Loading VAE: {cfg['vae']}", flush=True)
        vae = AutoencoderKL.from_pretrained(cfg['vae']).eval().to(self.device)
        self.models = model, create_diffusion(str(cfg['diffusion_steps'])), vae
        print('Loading LPIPS', flush=True)
        self.metric = lpips.LPIPS(net='alex').eval().to(self.device)
        self.spacing = data_cfg[cfg['action_dataset']]['metric_waypoint_spacing']
        self.minimum = np.array(data_cfg['action_stats']['min'])
        self.maximum = np.array(data_cfg['action_stats']['max'])
        self.rng = np.random.default_rng(cfg['seed'])
        self.planner = cfg.get('planner', 'discrete')
        self.last_plan = None
        if self.planner not in ('discrete', 'gaussian'):
            raise ValueError('planner must be discrete or gaussian')
        if self.planner == 'gaussian':
            if cfg['cem_iterations'] != 1:
                raise ValueError('Source-compatible Gaussian mode currently requires cem_iterations=1')
            if cfg.get('num_repeat_eval', 1) < 1:
                raise ValueError('num_repeat_eval must be positive')
            prior_file = cfg.get('action_prior', 'config/data_hyperparams_plan.yaml')
            self.prior = yaml.safe_load(Path(prior_file).read_text())[cfg['action_dataset']]


    def reset(self, rgb, goal):
        frame = transform(Image.fromarray(rgb)).to(self.device)
        self.history = frame[None].repeat(self.context, 1, 1, 1)
        self.goal = transform(Image.fromarray(goal)).to(self.device)[None]

    def observe(self, rgb):
        frame = transform(Image.fromarray(rgb)).to(self.device)
        self.history = torch.cat([self.history[1:], frame[None]])

    @torch.no_grad()
    def predict(self, history, actions):
        """Predict one transition for each history; shared by probe and planner."""
        a = np.asarray(actions)
        if a.ndim != 1 or len(a) != history.shape[0] or not np.isin(a, [0, 1, 2]).all():
            raise ValueError('One forward/left/right action is required per history')
        xy = np.zeros((len(a), 2))
        xy[a == 0, 0] = self.cfg['forward_m'] / self.spacing
        xy = 2 * (xy - self.minimum) / (self.maximum - self.minimum) - 1
        yaw = np.where(a == 1, 1., np.where(a == 2, -1., 0.)) * np.deg2rad(self.cfg['turn_deg'])
        condition = torch.tensor(np.column_stack([xy, yaw]), dtype=torch.float32, device=self.device)[:, None]
        return model_forward_wrapper(self.models, history, condition,
            self.cfg['frames_per_action'], self.latent, self.device, self.context)

    @torch.no_grad()
    def score(self, sequences):
        outputs = []
        for batch in range(0, len(sequences), self.cfg['batch_size']):
            actions = sequences[batch:batch + self.cfg['batch_size']]
            history = self.history[None].repeat(len(actions), 1, 1, 1, 1)
            for step in range(actions.shape[1]):
                image = self.predict(history, actions[:, step])
                history = torch.cat([history[:, 1:], image[:, None]], dim=1)
            outputs.extend(self.metric(image.float(), self.goal.expand_as(image).float()).flatten().cpu().tolist())
        return np.array(outputs)

    @torch.no_grad()
    def score_deltas(self, sequences):
        """Autoregressive rollout as in original planner, stride=1."""
        repeats = self.cfg.get('num_repeat_eval', 1)
        all_costs = []
        for repeat in range(repeats):
            costs = []
            for begin in range(0, len(sequences), self.cfg['batch_size']):
                actions = sequences[begin:begin + self.cfg['batch_size']]
                print(f'Gaussian rollout repeat {repeat+1}/{repeats}, candidates {begin+1}-{begin+len(actions)}/{len(sequences)}', flush=True)
                history = self.history[None].repeat(len(actions), 1, 1, 1, 1)
                for step in range(actions.shape[1]):
                    condition = torch.tensor(actions[:, step:step+1], dtype=torch.float32, device=self.device)
                    image = model_forward_wrapper(self.models, history, condition,
                        self.cfg['frames_per_action'], self.latent, self.device, self.context)
                    history = torch.cat([history[:, 1:], image[:, None]], dim=1)
                costs.extend(self.metric(image.float(), self.goal.expand_as(image).float()).flatten().cpu().tolist())
            all_costs.append(costs)
        return np.asarray(all_costs).mean(axis=0)

    def act(self):
        if self.planner == 'gaussian':
            sequence, cost, self.last_plan = gaussian_search(
                self.score_deltas, self.rng, self.prior['mu'], self.prior['var_scale'],
                self.minimum, self.maximum, self.cfg['horizon'], self.cfg['candidates'], self.cfg['elites'])
            first = sequence[0]
            xy = ((first[:2] + 1) / 2 * (self.maximum - self.minimum) + self.minimum) * self.spacing
            return dict(kind='local_delta', forward_m=float(xy[0]), left_m=float(xy[1]),
                        yaw_rad=float(first[2])), cost
        sequence, cost = search(self.score, self.rng, self.cfg['horizon'],
            self.cfg['candidates'], self.cfg['elites'], self.cfg['cem_iterations'])
        return int(sequence[0]), cost
