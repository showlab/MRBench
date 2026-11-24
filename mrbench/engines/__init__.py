"""Engine registry.  The Isaac engine is registered lazily: importing this
module never imports Isaac Sim."""

from __future__ import annotations

from ..core.registry import Registry
from .base import Engine
from .mock import MockEngine

ENGINES: Registry = Registry("engine")
ENGINES.register("mock", MockEngine)


def _isaac(**kwargs) -> Engine:
    from .isaac.engine import IsaacEngine

    return IsaacEngine(**kwargs)


ENGINES.register("isaac", _isaac)


def make_engine(name: str, **kwargs) -> Engine:
    return ENGINES.get(name)(**kwargs)
