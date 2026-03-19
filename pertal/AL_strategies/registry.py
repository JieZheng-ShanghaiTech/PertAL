from __future__ import annotations

from typing import Callable, Dict, TypeVar


T = TypeVar("T")

STRATEGY_FACTORY: Dict[str, type] = {}


def register_strategy(name: str) -> Callable[[T], T]:
    """
    Decorator to register a strategy implementation.

    This avoids hard-coded if/elif chains when adding new strategies.
    """

    def decorator(cls: T) -> T:
        STRATEGY_FACTORY[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


def StrategyFactory(name: str) -> type:
    cls = STRATEGY_FACTORY.get(name)
    if cls is None:
        available = ", ".join(sorted(STRATEGY_FACTORY.keys()))
        raise ValueError(f"Strategy '{name}' not found. Available: {available}")
    return cls

