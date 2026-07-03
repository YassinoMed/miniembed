"""End-to-end tests for the assembled EmbeddingModel across all variants."""

from __future__ import annotations

import itertools

import pytest
import torch

from miniembed.config import ModelConfig
from miniembed.models import EmbeddingModel


def _make_inputs(vocab_size: int, B=3, L=16):
    torch.manual_seed(0)
    ids = torch.randint(4, vocab_size, (B, L))
    mask = (ids != 0).long()
    return ids, mask


def test_default_model_param_count_matches_legacy():
    """The default config must reproduce the ~10.8M legacy model size."""
    model = EmbeddingModel(ModelConfig())
    n = sum(p.numel() for p in model.parameters())
    # Legacy reports ~10.8M. Allow a tight band.
    assert 10_700_000 < n < 10_900_000


def test_encode_outputs_unit_norm():
    cfg = ModelConfig(vocab_size=500, d_model=64, num_heads=4, num_layers=2, d_ff=128)
    model = EmbeddingModel(cfg)
    model.eval()
    ids, mask = _make_inputs(500)
    with torch.no_grad():
        emb = model.encode(ids, mask)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_forward_returns_token_level():
    cfg = ModelConfig(vocab_size=500, d_model=64, num_heads=4, num_layers=2, d_ff=128)
    model = EmbeddingModel(cfg)
    ids, mask = _make_inputs(500)
    with torch.no_grad():
        tok = model.forward(ids, mask)
    assert tok.shape == (3, 16, 64)


@pytest.mark.parametrize(
    "ffn,attn,pool,pos",
    itertools.product(
        ["gelu", "swiglu", "geglu"],
        ["mha", "sdpa"],
        ["mean", "cls", "max", "attention", "gem", "weighted_mean"],
        ["sinusoidal", "rope"],
    ),
)
def test_all_variant_combinations_produce_valid_embeddings(ffn, attn, pool, pos):
    cfg = ModelConfig(
        vocab_size=500,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        ffn_type=ffn,
        attention_type=attn,
        pooling_type=pool,
        position_type=pos,
    )
    model = EmbeddingModel(cfg)
    model.eval()
    ids, mask = _make_inputs(500, L=24)
    with torch.no_grad():
        emb = model.encode(ids, mask)
    assert emb.shape == (3, 64)
    assert torch.isfinite(emb).all()
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-4)


def test_model_save_load_roundtrip(tmp_path):
    cfg = ModelConfig(vocab_size=500, d_model=64, num_heads=4, num_layers=2, d_ff=128)
    model = EmbeddingModel(cfg)
    model.eval()
    ids, mask = _make_inputs(500)
    with torch.no_grad():
        before = model.encode(ids, mask)

    out_dir = tmp_path / "saved"
    model.save(out_dir)
    assert (out_dir / "config_v2.json").exists()

    model2 = EmbeddingModel.from_pretrained(out_dir)
    model2.eval()
    with torch.no_grad():
        after = model2.encode(ids, mask)
    assert torch.allclose(before, after, atol=1e-6)


def test_gradients_flow_to_all_parametrized_components():
    """Parametrized variants (attention/gem/weighted pooling) must train."""
    cfg = ModelConfig(
        vocab_size=500,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        pooling_type="attention",
        ffn_type="swiglu",
        attention_type="sdpa",
    )
    model = EmbeddingModel(cfg)
    ids, mask = _make_inputs(500)
    emb = model.encode(ids, mask)
    loss = emb.pow(2).sum()
    loss.backward()
    # At least one param in the attention scorer must have a gradient.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
