---
language: en
license: mit
tags:
  - text-embedding
  - sentence-similarity
  - semantic-search
  - product-matching
  - transformer
  - pytorch
  - from-scratch
library_name: pytorch
pipeline_tag: sentence-similarity
model-index:
  - name: MiniEmbed-Mini
    results: []
---

# MiniEmbed: Tiny, Powerful Embedding Models from Scratch

**MiniEmbed** is an ultra-compact text embedding model (Bi-Encoder) built entirely from scratch in PyTorch. No HuggingFace Transformers, no pre-trained weights -- just pure PyTorch.

**GitHub:** [github.com/bhandarisuraz/miniembed](https://github.com/bhandarisuraz/miniembed) (full repo with examples, tests, interactive demo, and documentation)

| Spec | Value |
|---|---|
| Parameters | ~10.8M |
| Model Size | ~42 MB |
| Embedding Dim | 256 |
| Vocab Size | 30,000 |
| Max Seq Length | 128 tokens |
| Architecture | 4-layer Transformer Encoder |
| Pooling | Mean Pooling + L2 Normalization |
| Training Loss | MNRL (Multiple Negatives Ranking Loss) |
| Training Data | ~3.8M pairs (NQ, GooAQ, MSMARCO, WDC, ECInstruct) |

## Quick Start

```bash
pip install torch numpy scikit-learn huggingface_hub
```

```python
from huggingface_hub import snapshot_download

# Download model (one-time)
model_dir = snapshot_download("surazbhandari/miniembed")

# Add src to path
import sys
sys.path.insert(0, model_dir)

from src.inference import EmbeddingInference

# Load -- just like sentence-transformers!
model = EmbeddingInference.from_pretrained("surazbhandari/miniembed")

# Similarity
score = model.similarity("Machine learning is great", "AI is wonderful")
print(f"Similarity: {score:.4f}")  # 0.4287

# Semantic Search
docs = ["Python is great for AI", "I love pizza", "Neural networks learn patterns"]
results = model.search("deep learning frameworks", docs, top_k=2)
for r in results:
    print(f"  [{r['score']:.3f}] {r['text']}")
# [0.498] Neural networks learn patterns
# [0.413] Python is great for AI

# Clustering
result = model.cluster_texts(["ML is cool", "Pizza is food", "AI rocks"], n_clusters=2)
# Cluster 1: ['Pizza is food']
# Cluster 2: ['ML is cool', 'AI rocks']
```

## Also Available via GitHub

```bash
git clone https://github.com/bhandarisuraz/miniembed.git
cd miniembed
pip install -r requirements.txt

python -c "
from src.inference import EmbeddingInference
model = EmbeddingInference.from_pretrained('models/mini')
print(model.similarity('hello world', 'hi there'))
"
```

## Capabilities

- **Semantic Search** -- Find meaning-based matches, not keyword overlap.
- **Re-Ranking** -- Sort candidates by true semantic relevance.
- **Clustering** -- Group texts into logical categories automatically.
- **Product Matching** -- Match items across platforms with messy titles.

## Architecture

Custom 4-layer Transformer encoder built from first principles:

- Token Embedding (30K vocab) + Sinusoidal Positional Encoding
- 4x Pre-LayerNorm Transformer Encoder Layers
- Multi-Head Self-Attention (4 heads, d_k=64)
- Position-wise Feed-Forward (GELU activation, d_ff=1024)
- Mean Pooling over non-padded tokens
- L2 Normalization (unit hypersphere projection)

## Training

Trained on ~3.8 million text pairs from public datasets:

| Dataset | Type |
|---|---|
| Natural Questions (NQ) | Q&A / General |
| GooAQ | Knowledge Search |
| WDC Product Matching | E-commerce |
| ECInstruct | E-commerce Tasks |
| MS MARCO | Web Search |

**Training details:**
- Training time: ~49 hours
- Final loss: 0.0748
- Optimizer: AdamW
- Batch size: 256

## Files

```
surazbhandari/miniembed
|-- README.md           # This model card
|-- config.json         # Architecture config
|-- model.safetensors   # Pre-trained weights (Safe & Fast)
|-- model.pt            # Pre-trained weights (Legacy PyTorch)
|-- tokenizer.json      # 30K word-level vocabulary
|-- training_info.json  # Training metadata
|-- src/
    |-- __init__.py
    |-- model.py        # Full architecture code
    |-- tokenizer.py    # Tokenizer implementation
    |-- inference.py    # High-level API (supports HF auto-download)
```

## Limitations

- Word-level tokenizer (no subword/BPE) -- unknown words map to [UNK]
- 128 token max sequence length
- Trained primarily on English text
- Best suited for short-form text (queries, product titles, sentences)

## v2 Platform (Phase 1)

The `miniembed/` package ships alongside this model card with a configurable
architecture: BPE/WordPiece/Unigram tokenizers, RoPE, SwiGLU/GEGLU FFNs, SDPA
attention, 6 pooling strategies. The default configuration is **weight-compatible**
with this checkpoint. See [`miniembed/README.md`](miniembed/README.md).

## Citation

```bibtex
@software{Bhandari_MiniEmbed_2026,
  author  = {Bhandari, Suraj},
  title   = {{MiniEmbed: Tiny, Powerful Embedding Models from Scratch}},
  url     = {https://github.com/bhandarisuraz/miniembed},
  version = {1.0.0},
  year    = {2026}
}
```

## License

MIT
