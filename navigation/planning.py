"""Categorical CEM over executable forward/left/right action sequences."""
import numpy as np


def search(score, rng, horizon=3, candidates=24, elites=6, iterations=3):
    if not (horizon > 0 and iterations > 0 and 1 <= elites <= candidates):
        raise ValueError('Require horizon/iterations > 0 and 1 <= elites <= candidates')
    probabilities = np.full((horizon, 3), 1 / 3)
    best, best_score = None, float('inf')
    for _ in range(iterations):
        actions = np.stack([rng.choice(3, candidates, p=p) for p in probabilities], axis=1)
        costs = np.asarray(score(actions))
        if costs.shape != (candidates,) or not np.isfinite(costs).all():
            raise ValueError('Scorer must return one finite cost per candidate')
        order = np.argsort(costs)
        if costs[order[0]] < best_score:
            best, best_score = actions[order[0]].copy(), float(costs[order[0]])
        selected = actions[order[:elites]]
        frequencies = np.stack([(selected == a).mean(axis=0) for a in range(3)], axis=1)
        probabilities = .9 * frequencies + .1 / 3
    return best, best_score
