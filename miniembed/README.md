# MiniEmbed v2 — Configurable Embedding Platform (Phase 1)

> **This is the evolving "platform" layer.** The original `src/` package and
> `models/mini` checkpoint remain unchanged and fully functional.

## What's new in Phase 1?

| Area | Detail |
|---|---|
| **Pluggable tokenizers** | BPE, WordPiece, Unigram (all from scratch) + WordLevel (legacy) + SentencePiece (lazy). |
| **RoPE** | Rotary Position Embedding as an alternative to sinusoidal. |
| **Gated FFN** | SwiGLU and GEGLU join the classic GELU FFN. |
| **SDPA attention** | Fused kernel backend (PyTorch ≥ 2.0) with automatic fallback. |
| **6 pooling strategies** | Mean, CLS, Max, Attention, GeM (learnable p), Weighted Mean. |
| **Config-driven model** | Every component is selectable from a single frozen dataclass. |
| **Weight-compatible** | The default config loads `models/mini` weights **bit-for-bit identical** to the legacy model. |
| **Test suite** | 137 tests covering tokenizers, all 72 architecture combos, compatibility, and legacy anti-regression. |

---

## Quick start

```python
from miniembed.models import EmbeddingModel, ModelConfig
from miniembed.tokenizers import get_tokenizer

# Default config = exactly the shipped models/mini architecture
model = EmbeddingModel(ModelConfig())

# Or go custom:
cfg = ModelConfig(
    d_model=256, num_heads=4, num_layers=4, d_ff=1024,
    position_type="rope",      # sinusoidal | rope
    ffn_type="swiglu",         # gelu | swiglu | geglu
    attention_type="sdpa",     # mha | sdpa
    pooling_type="gem",        # mean | cls | max | attention | gem | weighted_mean
)
model = EmbeddingModel(cfg)

# Encode
import torch
ids = torch.tensor([[2, 7, 8, 9, 3, 0, 0]])    # [CLS] tokens [SEP] [PAD] [PAD]
mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
embedding = model.encode(ids, mask)               # [1, 256], L2-normalized

# Load pretrained weights (identical output to src.inference)
model = EmbeddingModel.from_pretrained("models/mini")
```

### Tokenizers

```python
from miniembed.tokenizers import get_tokenizer, list_tokenizers

print(list_tokenizers())  # ['bpe', 'sentencepiece', 'unigram', 'wordlevel', 'wordpiece']

# Train a BPE tokenizer from scratch
tok = get_tokenizer("bpe", vocab_size=8000)
tok.train(my_corpus_texts, min_freq=2)
tok.save("my_bpe.json")

# Load and encode
from miniembed.tokenizers import load_tokenizer
tok = load_tokenizer("my_bpe.json")
out = tok.encode("hello world", max_length=64)
print(out.input_ids.shape)   # torch.Size([64])
print(out.attention_mask)    # tensor([1, 1, ..., 1, 0, ...])

# Batch encode
batch = tok.encode_batch(["hello world", "goodbye world"], max_length=64)
print(batch.input_ids.shape)  # torch.Size([2, 64])
```

---

## Architecture variants (72 tested combinations)

```
Position:   sinusoidal | rope
Attention:  mha (from-scratch) | sdpa (fused kernel, auto-fallback)
FFN:        gelu (classic) | swiglu | geglu (gated)
Pooling:    mean | cls | max | attention | gem | weighted_mean
```

Every combination produces valid, finite, unit-norm embeddings. All 72 are
tested in CI.

| Component | Config key | Options | Default |
|---|---|---|---|
| Positional encoding | `position_type` | `sinusoidal`, `rope` | `sinusoidal` |
| Attention | `attention_type` | `mha`, `sdpa` | `mha` |
| Feed-forward | `ffn_type` | `gelu`, `swiglu`, `geglu` | `gelu` |
| Pooling | `pooling_type` | `mean`, `cls`, `max`, `attention`, `gem`, `weighted_mean` | `mean` |
| GeM exponent | `pooling_p` | float | `3.0` |
| L2 normalize | `normalize` | bool | `True` |

---

## Package structure

```
miniembed/
├── __init__.py           # Public exports (lazy imports)
├── config.py             # ModelConfig, TokenizerConfig, TrainingConfig
├── registry.py           # @register / get() component registry
├── compat.py             # Legacy API re-export shim
├── core/
│   ├── types.py          # TokenizerOutput dataclass
├── tokenizers/
│   ├── base.py            # BaseTokenizer ABC
│   ├── wordlevel.py       # Legacy word-level (models/mini compatible)
│   ├── bpe.py             # BPE from scratch
│   ├── wordpiece.py       # WordPiece from scratch
│   ├── unigram.py         # Unigram + Viterbi from scratch
│   └── sentencepiece.py   # SentencePiece adapter (lazy import)
├── models/
│   ├── positional.py      # SinusoidalPE + RoPE
│   ├── attention.py       # MHA + SDPA (auto-fallback)
│   ├── feedforward.py     # GELU / SwiGLU / GEGLU
│   ├── pooling.py          # 6 strategies
│   ├── encoder.py          # Pre-LayerNorm encoder layer
│   └── embedding_model.py  # EmbeddingModel (assembler)
tests/
├── conftest.py
├── test_tokenizers.py
├── test_tokenizers_train.py
├── test_model_components.py
├── test_model_assembly.py
├── test_compatibility.py   # Embeddings identical to legacy
└── test_legacy_api.py      # Anti-regression
```

---

## Registry pattern

Add a custom component without touching core code:

```python
from miniembed.registry import register
from miniembed.models.feedforward import _GatedFFN
import torch.nn.functional as F

@register("ffn", "my_gated")
class MyGatedFFN(_GatedFFN):
    def _gate(self, x):
        return torch.sigmoid(x)  # custom gate activation

# Then use it from config:
cfg = ModelConfig(ffn_type="my_gated")
```

---

## Roadmap

Phase 1 (this release): Architecture modernization.
Phase 2–14: Training pipeline, benchmarks MTEB/BEIR, optimization (ONNX/TensorRT),
FastAPI, deployment (Docker/K8s), monitoring, CI/CD, MkDocs, demo v2, specialized
variants, and full documentation.

See the root [README.md](../README.md) for the complete roadmap.
