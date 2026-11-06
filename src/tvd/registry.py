"""A table from names to classes, so that a settings file can choose an implementation by its name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Implementations add themselves with ``@REGISTRY.register("name")``, and ``create`` builds one by its name.

    Adding an implementation never changes the code that uses it (the open/closed principle).
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._classes: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def add(cls: type[T]) -> type[T]:
            if name in self._classes:
                raise ValueError(f"A {self._kind} named {name!r} is already registered: {self._classes[name].__name__}")
            self._classes[name] = cls
            return cls

        return add

    def create(self, name: str, **options: Any) -> T:
        if name not in self._classes:
            raise ValueError(f"Unknown {self._kind} {name!r}. Available: {', '.join(self.names()) or 'none'}")
        return self._classes[name](**options)

    def names(self) -> list[str]:
        return sorted(self._classes)
