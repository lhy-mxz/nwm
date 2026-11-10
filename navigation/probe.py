"""One controlled predicted-versus-rendered transition, without CEM."""
import json
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from misc import transform


def pixels(tensor):
    return (tensor.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5)
            .permute(1, 2, 0).round().byte().numpy())


@torch.no_grad()
def run_probe(env, policy, initial, action_name, cfg, output):
    action = {'forward': 0, 'left': 1, 'right': 2}[action_name]
    print(f'Predicting one {action_name} transition ({cfg["diffusion_steps"]} diffusion steps)', flush=True)
    before = env.sim.get_agent(0).get_state()
    position_before = np.asarray(before.position).copy()
    torch.cuda.synchronize(policy.device)
    started = time.perf_counter()
    prediction = policy.predict(policy.history[None], np.array([action]))
    torch.cuda.synchronize(policy.device)
    elapsed = time.perf_counter() - started
    actual, collided = env.step(action)
    position_after = np.asarray(env.sim.get_agent(0).get_state().position).copy()
    actual_tensor = transform(Image.fromarray(actual)).to(policy.device)[None]
    lpips_error = float(policy.metric(prediction.float(), actual_tensor.float()).item())
    if not np.isfinite(lpips_error):
        raise ValueError('Non-finite predicted-versus-actual LPIPS')
    Image.fromarray(actual).save(output / 'actual_raw.png')
    frames = [pixels(policy.history[-1]), pixels(prediction[0]), pixels(actual_tensor[0])]
    names = ['initial_processed', 'predicted', 'actual_processed']
    for name, frame in zip(names, frames):
        Image.fromarray(frame).save(output / f'{name}.png')
    panel = Image.new('RGB', (3 * 224, 248), 'white')
    draw = ImageDraw.Draw(panel)
    for i, (name, frame) in enumerate(zip(names, frames)):
        panel.paste(Image.fromarray(frame), (i * 224, 24))
        draw.text((i * 224 + 4, 4), name, fill='black')
    panel.save(output / 'comparison.png')
    report = dict(mode='single-step', action=action_name, collided=collided,
                  position_before=position_before.tolist(), position_after=position_after.tolist(),
                  actual_translation_m=float(np.linalg.norm(position_after - position_before)),
                  prediction_seconds=elapsed, predicted_actual_lpips=lpips_error, config=cfg,
                  note='Repeated initial RGB history; one stochastic prediction; no navigation success claim. Collision can invalidate commanded-motion comparison.')
    (output / 'probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
