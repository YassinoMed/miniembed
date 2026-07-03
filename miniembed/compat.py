"""
Backward-compatibility shim.

Re-exports the legacy public API so that code written against the old
``src.inference`` module can keep working while new code uses the v2 surface.

This module does *not* replace ``src`` (kept strictly intact per the
retro-compatibility requirement); it merely offers a second, stable import
path:

>>> from miniembed.compat import EmbeddingInference  # legacy API, v2 plumbing

For now both point to the same battle-tested legacy implementation; the
inference layer will be migrated to the v2 model in a later phase once the
weights compatibility is empirically validated end-to-end on the demo.
"""

from __future__ import annotations


def __getattr__(name: str):  # PEP 562 lazy module-level getattr
    # Import lazily so that a missing optional dep doesn't break package import.
    if name in {"EmbeddingInference", "EmbeddingModelManager"}:
        try:
            from src.inference import (  # type: ignore
                EmbeddingInference,
                EmbeddingModelManager,
            )

            return {"EmbeddingInference": EmbeddingInference, "EmbeddingModelManager": EmbeddingModelManager}[name]
        except ImportError:
            raise ImportError(
                "compat shim requires the legacy 'src' package on sys.path."
            )
    raise AttributeError(f"module 'miniembed.compat' has no attribute {name!r}")


__all__ = ["EmbeddingInference", "EmbeddingModelManager"]
