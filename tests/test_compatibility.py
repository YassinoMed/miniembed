"""Backward-compatibility test: new model must reproduce legacy embeddings.

This is the strongest guarantee of Phase 1: loading ``models/mini`` into the
new :class:`EmbeddingModel` must produce embeddings identical to the legacy
``src.inference.EmbeddingInference``, because the new default config reproduces
the exact legacy architecture (sinusoidal + MHA + GELU + mean pooling) and
shares the same weight-key names.

Skipped automatically when the pretrained ``models/mini`` is absent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch.manual_seed(42)

TEXTS = [
    "Machine learning is great",
    "AI is wonderful",
    "Python is great for AI",
    "Neural networks learn patterns",
    "I love pizza",
]


@pytest.fixture(scope="module")
def legacy_model(models_mini_path):
    if not (models_mini_path / "model.safetensors").exists() and not (
        models_mini_path / "model.pt"
    ).exists():
        pytest.skip("models/mini weights not found")
    from src.inference import EmbeddingInference

    return EmbeddingInference.from_pretrained(str(models_mini_path), device="cpu")


@pytest.fixture(scope="module")
def new_model(models_mini_path):
    if not (models_mini_path / "model.safetensors").exists() and not (
        models_mini_path / "model.pt"
    ).exists():
        pytest.skip("models/mini weights not found")
    from miniembed.models import EmbeddingModel

    return EmbeddingModel.from_pretrained(str(models_mini_path))


def test_legacy_and_new_embeddings_match(legacy_model, new_model):
    legacy_emb = legacy_model.encode(TEXTS)  # numpy [N, 256]

    # Tokenize with the legacy tokenizer to guarantee identical input.
    enc = [legacy_model.tokenizer.encode(t, 64) for t in TEXTS]
    ids = torch.stack([e["input_ids"] for e in enc])
    mask = torch.stack([e["attention_mask"] for e in enc])
    with torch.no_grad():
        new_emb = new_model.encode(ids, mask).numpy()

    assert new_emb.shape == legacy_emb.shape == (len(TEXTS), 256)
    max_diff = float(np.max(np.abs(legacy_emb - new_emb)))
    # Identical architecture + same weights => numerically equal.
    assert max_diff < 1e-5, f"Embeddings differ by {max_diff}"


def test_new_model_loads_strict_state_dict(models_mini_path):
    """Loading must succeed with strict=True (no missing/unexpected keys)."""
    from miniembed.models import EmbeddingModel

    model = EmbeddingModel.from_pretrained(str(models_mini_path))
    # Re-load to confirm strict loading doesn't raise.
    if (models_mini_path / "model.safetensors").exists():
        from safetensors.torch import load_file

        sd = load_file(str(models_mini_path / "model.safetensors"))
    else:
        sd = torch.load(
            models_mini_path / "model.pt", map_location="cpu", weights_only=True
        )
    model.load_state_dict(sd, strict=True)  # must not raise


def test_similarity_is_preserved(legacy_model, new_model):
    """Pairwise similarity ranking must be identical."""
    from src.inference import EmbeddingInference  # noqa: F401

    legacy_sim = legacy_model.similarity(TEXTS[0], TEXTS[1])

    enc0 = legacy_model.tokenizer.encode(TEXTS[0], 64)
    enc1 = legacy_model.tokenizer.encode(TEXTS[1], 64)
    ids = torch.stack([enc0["input_ids"], enc1["input_ids"]])
    mask = torch.stack([enc0["attention_mask"], enc1["attention_mask"]])
    with torch.no_grad():
        emb = new_model.encode(ids, mask).numpy()
    new_sim = float(np.dot(emb[0], emb[1]))

    assert abs(legacy_sim - new_sim) < 1e-5
