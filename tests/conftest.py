"""Shared pytest fixtures for the MiniEmbed test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the repo root (containing both `miniembed` and the legacy `src`) is
# importable regardless of where pytest is invoked from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Text fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "Machine learning is great"


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture
def small_corpus() -> list[str]:
    """A small, deterministic corpus used to train tokenizers in tests."""
    return [
        "machine learning is great",
        "machine learning models",
        "deep learning frameworks",
        "neural networks learn patterns",
        "python is great for ai",
        "i love pizza and pasta",
        "pizza is delicious food",
        "cats and dogs are pets",
        "the cat sat on the mat",
        "neural networks are powerful",
    ] * 6  # repeat so frequencies are non-trivial


@pytest.fixture
def two_texts() -> list[str]:
    return ["machine learning is great", "AI is wonderful"]


# ---------------------------------------------------------------------------
# Model config fixtures (tiny model for fast tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config_kwargs() -> dict:
    """Kwargs producing a fast, tiny model."""
    return dict(
        vocab_size=500,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
    )


@pytest.fixture(scope="session")
def models_mini_path() -> Path:
    """Path to the shipped pretrained model (skipped if absent)."""
    return ROOT / "models" / "mini"
