"""
Centralized, strongly-typed configuration for MiniEmbed v2.

All configuration is expressed as frozen dataclasses so that:

* IDEs and type-checkers understand the schema;
* ``asdict``/``from_dict`` make (de)serialization to JSON trivial, which is
  required to save/load model checkpoints (see
  :class:`miniembed.models.embedding_model.EmbeddingModel`);
* defaults encode the *current* shipped architecture so that the pre-trained
  ``models/mini`` checkpoint can be loaded with zero configuration.

A :class:`ModelConfig` is intentionally a superset of the legacy ``src``
``config.json``: every extra field defaults to the value that reproduces the
original architecture, guaranteeing strict backward compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

#: Valid choices for the configurable components. Kept module-level so the
#: dataclass field metadata can reference them for validation/documentation.
POSITION_TYPES = ("sinusoidal", "rope")
ATTENTION_TYPES = ("mha", "sdpa")
FFN_TYPES = ("gelu", "swiglu", "geglu")
POOLING_TYPES = ("mean", "cls", "max", "attention", "gem", "weighted_mean")


@dataclass(frozen=True)
class ModelConfig:
    """Blueprint for assembling an :class:`EmbeddingModel`.

    Every field defaults to the value that reproduces the shipped ``mini``
    checkpoint (4-layer / 256-dim / 4-head / GELU / sinusoidal / mean-pool).
    """

    # --- architecture core ---
    vocab_size: int = 30000
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 128
    pad_token_id: int = 0
    dropout: float = 0.1

    # --- pluggable component selectors (Phase 1) ---
    position_type: str = "sinusoidal"
    attention_type: str = "mha"
    ffn_type: str = "gelu"
    pooling_type: str = "mean"

    # --- component hyperparameters ---
    pooling_p: float = 3.0  # GeM pooling exponent (initial value, learnable)

    # --- normalization toggles ---
    normalize: bool = True  # L2-normalize final embedding

    def __post_init__(self) -> None:
        self._check("position_type", self.position_type, POSITION_TYPES)
        self._check("attention_type", self.attention_type, ATTENTION_TYPES)
        self._check("ffn_type", self.ffn_type, FFN_TYPES)
        self._check("pooling_type", self.pooling_type, POOLING_TYPES)
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads "
                f"({self.num_heads})."
            )
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

    @staticmethod
    def _check(field_name: str, value: str, choices: tuple[str, ...]) -> None:
        if value not in choices:
            raise ValueError(
                f"Invalid {field_name}={value!r}. Expected one of {choices}."
            )

    # --- (de)serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """Build a config from a dict, ignoring unknown keys (forward-compat)."""
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration describing which tokenizer to use and how to train it."""

    tokenizer_type: str = "wordlevel"
    vocab_size: int = 30000
    max_length: int = 128
    lowercase: bool = True
    # Special token ids are fixed for backward compatibility with the
    # pre-trained model (PAD=0, UNK=1, CLS=2, SEP=3).
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    # training-time hyperparameters (used by Phase 2)
    min_freq: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenizerConfig:
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# Placeholder for Phase 2 (training pipeline). Defined now so config.py is the
# single source of truth for all stages of the roadmap.
@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters (consumed starting Phase 2).

    Defined here for forward compatibility; the training pipeline is not part
    of Phase 1.
    """

    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    loss_type: str = "mnrl"  # mnrl | triplet | contrastive | cosine | classification
    temperature: float = 0.05
    precision: str = "fp32"  # fp32 | fp16 | bf16
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False


__all__ = [
    "ModelConfig",
    "TokenizerConfig",
    "TrainingConfig",
    "POSITION_TYPES",
    "ATTENTION_TYPES",
    "FFN_TYPES",
    "POOLING_TYPES",
]
