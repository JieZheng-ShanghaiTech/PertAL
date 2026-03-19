from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .base import Scorer, ScoringContext


class LLMScorer(Scorer):
    """
    LLM prior scorer (pre-computed offline).

    Expected artifact layout:
      <kernel_root>/<llm_name>_zcore/kernel.pkl
    where kernel.pkl is a 1D numpy array aligned to the perturbation universe.
    """

    def score(self, ctx: ScoringContext) -> np.ndarray:
        if ctx.llm_name is None:
            raise ValueError("LLMScorer requires ctx.llm_name but it is None.")

        if ctx.kernel_root is not None:
            llm_kernel_path = ctx.kernel_root / f"{ctx.llm_name}_zcore" / "kernel.pkl"
        else:
            # Backward-compatible fallback (repo-local ./data layout).
            dataset_name = getattr(ctx.strategy, "dataset_name", None)
            if dataset_name is None:
                raise ValueError(
                    "LLMScorer requires ctx.kernel_root or ctx.strategy.dataset_name for fallback resolution."
                )
            base_path = Path(__file__).resolve().parents[2]
            llm_kernel_path = (
                base_path
                / "data"
                / f"{dataset_name}_kernels"
                / "knowledge_kernels_1k"
                / f"{ctx.llm_name}_zcore"
                / "kernel.pkl"
            )

        with open(llm_kernel_path, "rb") as f:
            llm_scores_all = pickle.load(f)

        if not isinstance(llm_scores_all, np.ndarray):
            llm_scores_all = np.asarray(llm_scores_all)

        return llm_scores_all[np.where(ctx.pool_mask)[0]]
