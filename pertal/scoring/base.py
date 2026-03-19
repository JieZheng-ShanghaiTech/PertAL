from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ScoringContext:
    """
    Context passed to scorers during a strategy query.

    We keep this intentionally lightweight and avoid importing heavy ML deps here.
    """

    pert_list: np.ndarray
    pool_mask: np.ndarray

    # Strategy/loader needed by certain scorers (e.g. gradient scorer).
    strategy: Any
    train_loader: Any

    # LLM prior
    kernel_root: Optional[Path]
    llm_name: Optional[str]

    # Gradient scorer
    gradient_weight: float


class Scorer(ABC):
    """
    Base scorer interface.

    Implementations must return a 1D numpy array of length equal to the pool size
    (i.e. `pool_mask.sum()`), aligned to the pool ordering induced by `pert_list`
    and `pool_mask`.
    """

    @abstractmethod
    def score(self, ctx: ScoringContext) -> np.ndarray:
        raise NotImplementedError

