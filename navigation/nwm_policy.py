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


class NWMPolicy:
    def __init__(self, cfg, model_cfg, data_cfg):
        if not torch.cuda.is_available():
            raise RuntimeError('NWM Habitat navigation requires CUDA')
        self.cfg, self.device = cfg, f"cuda:{cfg['gpu']}"
        self.context = model_cfg['context_size']
        self.latent = model_cfg['image_size'] // 8
        if model_cfg['image_size'] != 224:
            raise ValueError('Current image preprocessing requires image_size=224')
        model = CDiT_models[model_cfg['model']](input_size=self.latent, context_size=self.context)
        checkpoint = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['ema'], strict=True)
        model.eval().to(self.device)
        vae = AutoencoderKL.from_pretrained(cfg['vae']).eval().to(self.device)
        self.models = model, create_diffusion(str(cfg['diffusion_steps'])), vae
        self.metric = lpips.LPIPS(net='alex').eval().to(self.device)
        self.spacing = data_cfg[cfg['action_dataset']]['metric_waypoint_spacing']
        self.minimum = np.array(data_cfg['action_stats']['min'])
        self.maximum = np.array(data_cfg['action_stats']['max'])
        self.rng = np.random.default_rng(cfg['seed'])

    def reset(self, rgb, goal):
        frame = transform(Image.fromarray(rgb)).to(self.device)
        self.history = frame[None].repeat(self.context, 1, 1, 1)
        self.goal = transform(Image.fromarray(goal)).to(self.device)[None]

    def observe(self, rgb):
        frame = transform(Image.fromarray(rgb)).to(self.device)
        self.history = torch.cat([self.history[1:], frame[None]])

    @torch.no_grad()
    def score(self, sequences):
        outputs = []
        for batch in range(0, len(sequences), self.cfg['batch_size']):
            actions = sequences[batch:batch + self.cfg['batch_size']]
            history = self.history[None].repeat(len(actions), 1, 1, 1, 1)
            for step in range(actions.shape[1]):
                a = actions[:, step]
                xy = np.zeros((len(a), 2))
                xy[a == 0, 0] = self.cfg['forward_m'] / self.spacing
                xy = 2 * (xy - self.minimum) / (self.maximum - self.minimum) - 1
                yaw = np.where(a == 1, 1., np.where(a == 2, -1., 0.)) * np.deg2rad(self.cfg['turn_deg'])
                condition = torch.tensor(np.column_stack([xy, yaw]), dtype=torch.float32, device=self.device)[:, None]
                image = model_forward_wrapper(self.models, history, condition,
                    self.cfg['frames_per_action'], self.latent, self.device, self.context)
                history = torch.cat([history[:, 1:], image[:, None]], dim=1)
            outputs.extend(self.metric(image.float(), self.goal.expand_as(image).float()).flatten().cpu().tolist())
        return np.array(outputs)

    def act(self):
        sequence, cost = search(self.score, self.rng, self.cfg['horizon'],
            self.cfg['candidates'], self.cfg['elites'], self.cfg['cem_iterations'])
        return int(sequence[0]), cost
