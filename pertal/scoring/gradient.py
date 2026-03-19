from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np

from .base import Scorer, ScoringContext


def _mean_by_key(keys: np.ndarray, values: np.ndarray) -> Dict[str, np.ndarray]:
    sums = defaultdict(lambda: None)
    counts = defaultdict(int)
    for k, v in zip(keys, values):
        if sums[k] is None:
            sums[k] = np.asarray(v, dtype=np.float64)
        else:
            sums[k] += v
        counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums}


class GradientScorer(Scorer):
    """
    Gradient sensitivity scorer used by PertAL.

    The score is computed as the sum of the top-K absolute input-gradient entries
    (K = gradient_dim * gradient_weight) per perturbation.
    """

    def score(self, ctx: ScoringContext) -> np.ndarray:
        grad = ctx.strategy.get_gradient_new(ctx.train_loader)
        pert2grad = _mean_by_key(grad["pert_cat"], grad["gradients"])

        grad_dim = grad["gradients"].shape[1]
        K = int(grad_dim * ctx.gradient_weight)
        if K <= 0:
            raise ValueError(f"gradient_weight={ctx.gradient_weight} yields K={K}, which is invalid.")

        pert2score = {}
        for pert, grad_vec in pert2grad.items():
            abs_grad = np.abs(grad_vec)
            top_k_indices = np.argsort(abs_grad)[-K:]
            pert2score[pert] = abs_grad[top_k_indices].sum()

        scores_all = np.stack([pert2score[i] for i in ctx.pert_list])
        return scores_all[np.where(ctx.pool_mask)[0]]

