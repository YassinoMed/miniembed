"""
Mini-Transformer Embedding Model
====================================
A lightweight transformer encoder for generating text embeddings.
Built from scratch using PyTorch.

Architecture:
- Token Embeddings + Sinusoidal Positional Encoding
- N Transformer Encoder Layers (Pre-LayerNorm)
- Multi-Head Self-Attention
- Position-wise Feed-Forward Networks
- Mean Pooling + L2 Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".
    
    Adds position information to token embeddings using sin/cos functions
    at different frequencies, allowing the model to understand token order.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute division term for frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional mask [batch_size, seq_len]
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x)  # [batch, seq, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape to [batch, num_heads, seq, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: [batch, num_heads, seq, seq]
        
        # Apply attention mask (for padding)
        if attention_mask is not None:
            # Expand mask: [batch, 1, 1, seq]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        # context: [batch, num_heads, seq, d_k]
        
        # Reshape back: [batch, seq, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Two linear transformations with a GELU activation in between.
    Applied to each position separately and identically.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer with Pre-LayerNorm.
    
    Components:
    1. Multi-Head Self-Attention with residual connection
    2. Position-wise Feed-Forward with residual connection
    
    Uses Pre-LayerNorm for better training stability.
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Sub-layers
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional mask [batch_size, seq_len]
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-norm attention block
        normed = self.norm1(x)
        attn_output = self.attention(normed, attention_mask)
        x = x + self.dropout(attn_output)  # Residual connection
        
        # Pre-norm feed-forward block
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)  # Residual connection
        
        return x


class MiniTransformerEmbedding(nn.Module):
    """
    Mini-Transformer Embedding Model.
    
    Converts variable-length text sequences into fixed-size dense vectors
    suitable for semantic similarity, search, and clustering tasks.
    
    Architecture:
    1. Token Embedding Layer (vocab → d_model)
    2. Sinusoidal Positional Encoding
    3. N Transformer Encoder Layers
    4. Mean Pooling (sequence → single vector)
    5. L2 Normalization (for cosine similarity)
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        
        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model, max_seq_len, dropout
        )
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Mask for padding [batch_size, seq_len]
            
        Returns:
            Token-level representations [batch_size, seq_len, d_model]
        """
        # Token embeddings with scaling
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        return x
    
    def encode(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode input tokens to a single embedding vector per sequence.
        
        Uses mean pooling over non-padded tokens, followed by L2 normalization.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Mask for padding [batch_size, seq_len]
            
        Returns:
            Normalized embeddings [batch_size, d_model]
        """
        # Get token-level representations
        token_embeddings = self.forward(input_ids, attention_mask)
        
        # Mean pooling
        if attention_mask is not None:
            # Expand mask for broadcasting: [batch, seq, 1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            
            # Sum of embeddings (masked)
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            
            # Count of non-padded tokens
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            # Mean
            embeddings = sum_embeddings / sum_mask
        else:
            # Simple mean over all tokens
            embeddings = torch.mean(token_embeddings, dim=1)
        
        # L2 normalization for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
