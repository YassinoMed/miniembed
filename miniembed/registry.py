"""
Lightweight component registry.

Components (tokenizers, FFN variants, pooling strategies, ...) register
themselves under a string name via :func:`register` and are later resolved by
name via :func:`get`. This keeps the selection logic free of ``if/elif``
chains and makes the framework trivially extensible: a downstream user can
``register("my_ffn")`` and pick it from config without touching core code.

Each registry is namespaced by *kind* (e.g. ``"tokenizer"``, ``"ffn"``) so
that names may collide across kinds without ambiguity.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

# kind -> {name -> factory (class or callable)}
_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register(kind: str, name: str) -> Callable[[Any], Any]:
    """Decorator registering ``obj`` under ``name`` in the ``kind`` registry.

    Parameters
    ----------
    kind : str
        Registry namespace, e.g. ``"tokenizer"``, ``"ffn"``, ``"pooling"``.
    name : str
        Public name used in config files to select the component.

    Returns
    -------
    Callable
        Class/callable decorator that returns the object unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered for ``kind``.
    """

    def decorator(obj: Any) -> Any:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket:
            raise ValueError(
                f"A {kind!r} component named {name!r} is already registered "
                f"({bucket[name]!r}). Pick a different name."
            )
        bucket[name] = obj
        return obj

    return decorator


def get(kind: str, name: str) -> Any:
    """Resolve a registered component.

    Parameters
    ----------
    kind : str
        Registry namespace.
    name : str
        Registered component name.

    Raises
    ------
    KeyError
        If ``kind`` or ``name`` is unknown (with a helpful message listing
        the available options).
    """
    bucket = _REGISTRY.get(kind)
    if bucket is None or name not in bucket:
        available = sorted((_REGISTRY.get(kind) or {}).keys())
        raise KeyError(
            f"No {kind!r} component named {name!r}. "
            f"Available: {available or 'none'}"
        )
    return bucket[name]


def available(kind: str) -> list[str]:
    """Return the sorted list of registered names for ``kind``."""
    return sorted((_REGISTRY.get(kind) or {}).keys())


def _clear() -> None:
    """Empty every registry. Intended for tests only."""
    _REGISTRY.clear()


__all__ = ["register", "get", "available", "_clear"]
