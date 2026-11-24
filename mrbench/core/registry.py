"""A tiny name -> object registry used for engines and policies."""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

from .errors import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: Optional[T] = None):
        """``REG.register("name", obj)`` or ``@REG.register("name")``."""
        if item is not None:
            self._add(name, item)
            return item

        def deco(obj: T) -> T:
            self._add(name, obj)
            return obj

        return deco

    def _add(self, name: str, item: T) -> None:
        if name in self._items:
            raise ConfigError(f"{self.kind} {name!r} is already registered")
        self._items[name] = item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise ConfigError(f"unknown {self.kind} {name!r}; available: {', '.join(self.names())}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

