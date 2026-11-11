"""Source-compatible single-round Gaussian planning, independent of Habitat.

Mirrors planning_eval.py: constant normalized XY increments, heading-derived
initial yaw, final yaw perturbation, elite mean of final deltas, final rollout.
The source's final-yaw / pi convention is intentionally preserved, not corrected.
"""
import numpy as np


def make_deltas(parameters, horizon, minimum, maximum):
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 3 or horizon < 1:
        raise ValueError('Expected [batch, 3] parameters and positive horizon')
    xy = np.repeat(parameters[:, None, :2], horizon, axis=1)
    physical = (xy + 1) / 2 * (maximum - minimum) + minimum
    headings = np.arctan2(physical[..., 1], physical[..., 0])
    yaw = np.diff(np.concatenate([np.zeros((len(parameters), 1)), headings], axis=1), axis=1)
    yaw[:, -1] += parameters[:, -1] * np.pi
    return np.concatenate([xy, yaw[..., None]], axis=-1)


def gaussian_search(score, rng, mu, sigma, minimum, maximum,
                    horizon=8, candidates=12, elites=5):
    if not 2 <= elites <= candidates or horizon < 1:
        raise ValueError('Gaussian search requires 2 <= elites <= candidates and horizon >= 1')
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    if mu.shape != (3,) or sigma.shape != (3,) or not np.isfinite([mu, sigma]).all() or (sigma < 0).any():
        raise ValueError('Prior must contain three finite means and nonnegative standard deviations')
    samples = rng.standard_normal((candidates, 3)) * sigma + mu
    deltas = make_deltas(samples, horizon, minimum, maximum)
    costs = np.asarray(score(deltas))
    if costs.shape != (candidates,) or not np.isfinite(costs).all():
        raise ValueError('Invalid candidate costs')
    selected = np.argsort(costs)[:elites]
    # Exactly the source update: elite final deltas, not original parameters.
    elite_actions = deltas[selected, -1]
    updated_mu = elite_actions.mean(axis=0)
    updated_sigma = elite_actions.std(axis=0, ddof=1)
    final = make_deltas(updated_mu[None], horizon, minimum, maximum)
    final_cost = np.asarray(score(final))
    if final_cost.shape != (1,) or not np.isfinite(final_cost).all():
        raise ValueError('Invalid final rollout cost')
    diagnostics = dict(prior_mu=mu.tolist(), prior_sigma=sigma.tolist(),
                       samples=samples.tolist(), costs=costs.tolist(), elite_indices=selected.tolist(),
                       updated_mu=updated_mu.tolist(), updated_sigma=updated_sigma.tolist(),
                       final_deltas=final[0].tolist(), final_cost=float(final_cost[0]))
    return final[0], float(final_cost[0]), diagnostics


def local_to_world(forward, left, yaw):
    """Habitat: +Y up, yaw=0 faces -Z; NWM local +Y points left."""
    return np.array([-np.sin(yaw) * forward - np.cos(yaw) * left,
                     0., -np.cos(yaw) * forward + np.sin(yaw) * left])
