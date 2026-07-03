"""Anti-regression test: the legacy ``src`` API must keep working unchanged.

Phase 1 must not touch ``src/`` at all. This test imports the legacy entry
points and exercises them end-to-end so any accidental breakage is caught.
Skipped when the pretrained weights are missing.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def legacy_inf(models_mini_path):
    if not (models_mini_path / "model.safetensors").exists() and not (
        models_mini_path / "model.pt"
    ).exists():
        pytest.skip("models/mini weights not found")
    from src.inference import EmbeddingInference

    return EmbeddingInference.from_pretrained(str(models_mini_path))


def test_legacy_imports():
    from src.inference import EmbeddingInference, EmbeddingModelManager  # noqa: F401
    from src.model import MiniTransformerEmbedding  # noqa: F401
    from src.tokenizer import SimpleTokenizer  # noqa: F401


def test_legacy_encode(legacy_inf):
    emb = legacy_inf.encode("hello world")
    assert emb.shape == (1, 256)


def test_legacy_similarity(legacy_inf):
    score = legacy_inf.similarity("machine learning", "artificial intelligence")
    assert -1.0 <= score <= 1.0


def test_legacy_search(legacy_inf):
    docs = ["Python is great for AI", "I love pizza", "Neural networks learn patterns"]
    results = legacy_inf.search("deep learning", docs, top_k=2)
    assert len(results) == 2
    assert all("score" in r and "text" in r for r in results)
    # Results should be sorted by descending score.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_legacy_cluster(legacy_inf):
    texts = ["ML is cool", "Pizza is food", "AI rocks", "Pasta is delicious"]
    result = legacy_inf.cluster_texts(texts, n_clusters=2)
    assert "labels" in result
    assert len(result["labels"]) == 4


@pytest.mark.xfail(
    reason="scipy native lib crash on this macOS/arm64 Python 3.12 environment",
    strict=False,
)
def test_legacy_cluster_xfail(legacy_inf):
    """Re-run to confirm the failure is environmental, not code-level."""
    test_legacy_cluster(legacy_inf)


def test_legacy_list_models():
    from src.inference import EmbeddingModelManager

    names = EmbeddingModelManager.list_models("models")
    assert "mini" in names
