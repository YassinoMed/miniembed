"""
Config-driven embedding model (Phase 1 architecture).

:class:`EmbeddingModel` assembles the pluggable components (positional,
attention, FFN, pooling) selected by a :class:`~miniembed.config.ModelConfig`
into a complete Bi-Encoder. The public surface mirrors the legacy model so the
high-level inference API is unchanged:

>>> from miniembed.models import EmbeddingModel, ModelConfig
>>> model = EmbeddingModel(ModelConfig())
>>> emb = model.encode(input_ids, attention_mask)  # [B, d_model], L2-normalized

Backward compatibility with ``models/mini``
-------------------------------------------
The default :class:`ModelConfig` reproduces the legacy architecture exactly
(word-level embeddings, sinusoidal positions, MHA, GELU FFN, mean pooling).
:meth:`from_pretrained` reads the legacy ``config.json`` + weights and verifies
that the reassembled model produces identical embeddings (see the
compatibility test). Weight-key names match the legacy module layout, so the
state dict loads with ``strict=True``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .attention import build_attention
from .feedforward import build_ffn
from .pooling import build_pooling
from .positional import RotaryEmbedding, build_positional


class EmbeddingModel(nn.Module):
    """Transformer Bi-Encoder producing L2-normalized embeddings.

    Parameters
    ----------
    config : ModelConfig
        Architecture blueprint. Defaults reproduce ``models/mini``.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.pad_token_id = config.pad_token_id

        # --- token embedding ---
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )

        # --- positional ---
        # rotary lives inside attention; sinusoidal is a stem module (or None).
        # Attribute is named ``positional_encoding`` (not ``positional``) so the
        # new model is byte-compatible with the legacy ``models/mini`` state
        # dict (which uses ``positional_encoding.pe``).
        self.positional_encoding = build_positional(
            config, config.d_model, config.max_seq_len, config.dropout
        )
        rotary = None
        if config.position_type == "rope":
            rotary = RotaryEmbedding(
                head_dim=config.d_model // config.num_heads,
                max_seq_len=config.max_seq_len,
            )
        self.rotary = rotary

        # --- encoder layers ---
        self.layers = nn.ModuleList(
            [
                self._build_layer(config)
                for _ in range(config.num_layers)
            ]
        )

        # --- final norm + pooling ---
        self.final_norm = nn.LayerNorm(config.d_model)
        self.pooling = build_pooling(
            config.pooling_type,
            config.d_model,
            max_seq_len=config.max_seq_len,
            pooling_p=config.pooling_p,
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _build_layer(self, config: ModelConfig):
        from .encoder import TransformerEncoderLayer

        attention = build_attention(
            config.attention_type,
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            rotary=self.rotary,
            max_seq_len=config.max_seq_len,
        )
        feed_forward = build_ffn(
            config.ffn_type, config.d_model, config.d_ff, dropout=config.dropout
        )
        return TransformerEncoderLayer(
            attention=attention,
            feed_forward=feed_forward,
            d_model=config.d_model,
            dropout=config.dropout,
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    # ------------------------------------------------------------------
    # Forward / encode
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Token-level representations ``[B, L, d_model]``."""
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        return x

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeddings ``[B, d_model]`` (mean pooling + L2 norm by default)."""
        token_embeddings = self.forward(input_ids, attention_mask)

        if attention_mask is None:
            attention_mask = (
                input_ids != self.pad_token_id
            ).long()

        pooled = self.pooling(token_embeddings, attention_mask)

        if self.config.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, model_dir: str | Path, state_dict_only: bool = False) -> None:
        """Save config + weights to ``model_dir``.

        Writes ``config_v2.json`` (the full :class:`ModelConfig`) and
        ``model.safetensors`` (if available) or ``model.pt``. The legacy
        ``config.json`` is left untouched when ``state_dict_only`` is False so
        the directory stays loadable by the old ``src`` code too.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        # New-format config (richer). Keep the name distinct from legacy.
        with open(model_dir / "config_v2.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        # Legacy-format config (so old loaders still work) when requested.
        if not state_dict_only:
            legacy = {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "d_ff": self.config.d_ff,
                "max_seq_len": self.config.max_seq_len,
                "pad_token_id": self.config.pad_token_id,
            }
            with open(model_dir / "config.json", "w") as f:
                json.dump(legacy, f, indent=2)

        try:
            from safetensors.torch import save_file

            save_file(self.state_dict(), str(model_dir / "model.safetensors"))
        except Exception:
            torch.save(self.state_dict(), model_dir / "model.pt")

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path, device: Optional[str] = None
    ) -> "EmbeddingModel":
        """Load a model from a directory.

        Accepts both the legacy ``models/mini`` layout (``config.json``) and
        the v2 layout (``config_v2.json``). Loads ``model.safetensors`` if
        present, else ``model.pt``.
        """
        model_dir = Path(model_dir)
        v2_path = model_dir / "config_v2.json"
        legacy_path = model_dir / "config.json"

        if v2_path.exists():
            with open(v2_path) as f:
                cfg = ModelConfig.from_dict(json.load(f))
        elif legacy_path.exists():
            with open(legacy_path) as f:
                cfg = ModelConfig.from_dict(json.load(f))
        else:
            raise FileNotFoundError(
                f"Neither config_v2.json nor config.json found in {model_dir}"
            )

        model = cls(cfg)

        st = model_dir / "model.safetensors"
        pt = model_dir / "model.pt"
        if st.exists():
            from safetensors.torch import load_file

            state_dict = load_file(str(st))
        elif pt.exists():
            state_dict = torch.load(pt, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No model.safetensors or model.pt in {model_dir}"
            )
        model.load_state_dict(state_dict)

        if device:
            model = model.to(device)
        model.eval()
        return model


__all__ = ["EmbeddingModel"]
