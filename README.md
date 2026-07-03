# MiniEmbed: Tiny, Powerful Embedding Models from Scratch

**MiniEmbed** is a research-grade toolkit for training and deploying ultra-compact text embedding models (Bi-Encoders) built entirely from scratch in PyTorch. While the industry chases billion-parameter giants, MiniEmbed proves that a **~42 MB / 10.8M parameter** model can deliver production-grade semantic intelligence for specialized domains.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/surazbhandari/miniembed)

---

## What Can MiniEmbed Do?

| Capability | Description |
|---|---|
| **Semantic Search** | Find meaning, not just keywords. Understands that *"kitten"* is similar to *"cat"*. |
| **Re-Ranking** | Sort candidates by true semantic relevance. Eliminates false positives. |
| **Clustering** | Group thousands of texts into logical categories automatically. |
| **Product Matching** | Match identical items across stores, even with messy or inconsistent titles. |
| **Text Encoding** | Convert any text into a dense 256-dimensional vector for downstream tasks. |

---

## Project Structure

```
miniembed/
|-- README.md               # You are here
|-- LICENSE                  # MIT License
|-- requirements.txt         # Python dependencies
|-- demo.py                  # Interactive Streamlit demo
|-- src/                     # Core library
|   |-- __init__.py
|   |-- model.py             # Transformer architecture (from scratch)
|   |-- tokenizer.py         # Custom word-level tokenizer
|   |-- inference.py         # High-level API for encoding & search
|-- models/
|   |-- mini/                # Pre-trained Mini model
|       |-- model.safetensors # Pre-trained weights (Safe & Fast)
|       |-- model.pt         # Pre-trained weights (Legacy)
|       |-- config.json      # Architecture blueprint
|       |-- tokenizer.json   # 30K vocabulary
|       |-- training_info.json  # Training metadata
|-- examples/                # Ready-to-run scripts
|   |-- basic_usage.py       # Encoding & similarity
|   |-- semantic_search.py   # Document retrieval
|   |-- clustering.py        # Text clustering with K-Means
|-- data/
    |-- sample_data.jsonl    # 10-pair demo dataset
```

> **Note:** Pre-trained weights (`model.safetensors` / `model.pt`, ~42 MB) are included in this repository. Clone and use immediately. `.safetensors` is recommended for security and faster loading.

---

## Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/bhandarisuraz/miniembed.git
cd miniembed
pip install -r requirements.txt
```

### 2. Use the Model

The pre-trained Mini model is included in `models/mini/`. Alternatively, you can load it directly from Hugging Face:

```python
from src.inference import EmbeddingInference

# Option A: From local files
model = EmbeddingInference.from_pretrained("models/mini")

# Option B: Direct from Hugging Face (auto-downloads)
model = EmbeddingInference.from_pretrained("surazbhandari/miniembed")
```

### 3. Try It Instantly
```python
from src.inference import EmbeddingInference

model = EmbeddingInference.from_pretrained("models/mini")

# Similarity
score = model.similarity("Machine learning is great", "AI is wonderful")
print(f"Similarity: {score:.4f}")  # 0.4287

# Semantic Search
docs = ["Python is great for AI", "I love pizza", "Neural networks learn patterns"]
results = model.search("deep learning frameworks", docs, top_k=2)
for r in results:
    print(f"  [{r['score']:.3f}] {r['text']}")
```

For full Hugging Face integration, ensure you have `huggingface_hub` installed:
```bash
pip install huggingface_hub
```

---

## Interactive Demo (`demo.py`)

A full-featured Streamlit dashboard for exploring the model's capabilities without writing code:

- **Similarity** -- Real-time cosine similarity between any two texts.
- **Semantic Search** -- Rank a custom document set against your query.
- **Clustering** -- Automatically categorize items using K-Means.
- **Text Encoding** -- Inspect raw 256-D vectors and their statistics.
- **CSV Matcher** -- Match records between two CSV files for deduplication or cross-platform product mapping.

```bash
streamlit run demo.py
```

---

## Architecture

MiniEmbed uses a **custom 4-layer Transformer encoder** built from scratch -- no HuggingFace, no pre-trained weights:

| Component | Specification |
|---|---|
| Embedding Dimension | 256 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Feed-Forward Dimension | 1,024 |
| Vocabulary Size | 30,000 |
| Max Sequence Length | 128 tokens |
| Total Parameters | ~10.8M |
| Model Size on Disk | ~42 MB |
| Pooling Strategy | Mean Pooling + L2 Normalization |

### Training Objective

Training uses **Multiple Negatives Ranking Loss (MNRL)**, the industry-standard contrastive objective for Bi-Encoders:

$$\mathcal{L} = -\sum_{i=1}^{n} \log \frac{e^{sim(q_i, p_i) / \tau}}{\sum_{j=1}^{n} e^{sim(q_i, p_j) / \tau}}$$

All embeddings are **L2-normalized**, projecting text onto a unit hypersphere where cosine similarity equals dot product -- enabling ultra-fast nearest-neighbor search.

---

## Training Data Sources

The pre-trained model was trained on ~3.8 million text pairs from the following open-source datasets:

| Dataset | Type | Source |
|---|---|---|
| **Natural Questions (NQ)** | Q&A / General | [HuggingFace](https://huggingface.co/datasets/google-research-datasets/natural_questions) |
| **GooAQ** | Knowledge Search | [HuggingFace](https://huggingface.co/datasets/sentence-transformers/gooaq) |
| **WDC Product Matching** | E-commerce | [HuggingFace](https://huggingface.co/datasets/wdc/products-2017) |
| **ECInstruct** | E-commerce Tasks | [HuggingFace](https://huggingface.co/datasets/NingLab/ECInstruct) |
| **MS MARCO** | Web Search | [HuggingFace](https://huggingface.co/datasets/microsoft/ms_marco) |

> **Legal Disclaimer**: These public datasets belong to their respective stakeholders and creators. Any copyright, licensing, or legal usage constraints must be consulted with the original authors individually.

---

## Performance

Results from the pre-trained Mini model:

| Metric | Value |
|---|---|
| **Training Loss** | 0.0748 (final) |
| **Training Samples** | 3,817,707 pairs |
| **Throughput** | ~1,000 samples/sec |
| **Encoding Latency** | ~3-5 ms per text |
| **Training Epochs** | 10 |

---

## Examples

Ready-to-run scripts in the `examples/` folder:

```bash
cd examples

# Basic encoding and similarity
python basic_usage.py

# Document retrieval
python semantic_search.py

# Text clustering with K-Means
python clustering.py
```

---

## Roadmap

- **mini-product** -- A further fine-tuned version of the Mini model, specialized for high-accuracy **product matching** is Coming soon...

---

## Citation

If you use MiniEmbed in your research, please cite:

```bibtex
@software{Bhandari_MiniEmbed_2026,
  author  = {Bhandari, Suraj},
  title   = {{MiniEmbed: Tiny, Powerful Embedding Models from Scratch}},
  url     = {https://github.com/bhandarisuraz/miniembed},
  version = {1.0.0},
  year    = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Explore, learn, and build smaller, smarter AI.

---

## 🚀 MiniEmbed v2 — Platform Evolution (Phase 1 shipped)

A new `miniembed/` package has been added alongside the original `src/`. It
provides a **configurable Transformer Bi-Encoder** with pluggable components:

| New capability | Details |
|---|---|
| **5 tokenizers** | BPE, WordPiece, Unigram (from scratch), WordLevel (legacy), SentencePiece (lazy) |
| **RoPE** | Rotary Position Embedding as alternative to sinusoidal |
| **Gated FFNs** | SwiGLU, GEGLU alongside classic GELU |
| **SDPA attention** | Fused kernel (PyTorch ≥ 2.0) with auto-fallback |
| **6 pooling strategies** | Mean, CLS, Max, Attention, GeM, Weighted Mean |
| **Config-driven** | Every variant selectable from a single `ModelConfig` dataclass |
| **100% compatible** | Default config loads `models/mini` weights **bit-for-bit identical** |
| **137 tests** | Covers all 72 architecture combinations + legacy anti-regression |

```python
# Load the existing pretrained model via the new API
from miniembed.models import EmbeddingModel
model = EmbeddingModel.from_pretrained("models/mini")

# Or build a custom architecture
from miniembed.models import ModelConfig
cfg = ModelConfig(position_type="rope", ffn_type="swiglu", pooling_type="gem")
model = EmbeddingModel(cfg)
```

See [`miniembed/README.md`](miniembed/README.md) for the full Phase 1 guide.
# miniembed
# miniembed
