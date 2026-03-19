"""
Scoring modules for active learning.

This package provides small, testable scoring components that can be composed by
strategies (e.g. PertAL) to compute candidate scores such as:
- gradient sensitivity
- LLM priors
"""

from .base import Scorer, ScoringContext
from .gradient import GradientScorer
from .llm import LLMScorer

__all__ = [
    "Scorer",
    "ScoringContext",
    "GradientScorer",
    "LLMScorer",
]

