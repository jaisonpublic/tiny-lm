"""Multi-Head Attention (MHA/GQA/MQA).

Multi-head attention allows the model to jointly attend to information from
different representation subspaces at different positions. This implementation
includes causal masking for autoregressive language modeling.

This module supports three attention layouts through ``n_kv_heads``:
- MHA: ``n_kv_heads == n_heads`` (GPT-style)
- GQA: ``1 < n_kv_heads < n_heads`` (LLaMA-style grouped-query attention)
- MQA: ``n_kv_heads == 1`` (multi-query attention)

Paper: "Attention is All You Need" (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762

Paper: "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019)
https://arxiv.org/abs/1911.02150

Paper: "GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints" (Ainslie et al., 2023)
https://arxiv.org/abs/2305.13245
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tiny_lm.model.position import apply_rotary_emb


class MultiHeadAttention(nn.Module):
    """Causal self-attention supporting MHA, GQA, and MQA.

    Splits the input into multiple heads, computes scaled dot-product attention
    for each head independently, then concatenates and projects the results.

    Uses ``F.scaled_dot_product_attention`` (PyTorch ≥ 2.0) which automatically
    dispatches to the fastest available backend (FlashAttention-2,
    Memory-Efficient Attention, or the math fallback).

    Args:
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        n_kv_heads: Number of key/value heads.
            - If None, defaults to n_heads (standard MHA)
            - If 1, uses MQA
            - If between 1 and n_heads, uses GQA
        context_length: Maximum sequence length
        dropout: Dropout probability for attention weights
        qkv_bias: Whether to use bias in Q, K, V projections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        # GQA: rep = how many times to repeat the KV heads to match Q heads?
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = dropout

        # Linear projections:
        # Q always uses n_heads, while K/V use n_kv_heads for GQA/MQA.
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            freqs_cis: Optional RoPE frequencies of shape (seq_len, head_dim // 2).
                If provided, applies rotary embedding to Q and K.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V: (batch_size, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape and split heads.
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Optional RoPE path (Mini-LLM style: frequencies are passed in).
        if freqs_cis is not None:
            Q, K = apply_rotary_emb(Q, K, freqs_cis)

        # For GQA/MQA, repeat K/V heads so they align with query heads.
        if self.n_rep > 1:
            K = K.repeat_interleave(self.n_rep, dim=2)
            V = V.repeat_interleave(self.n_rep, dim=2)

        # Transpose to (batch_size, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention with automatic backend selection
        # (FlashAttention-2, Memory-Efficient, or Math fallback).
        # is_causal=True applies the causal mask inside the fused kernel,
        # reducing memory usage from O(n²) to O(n).
        context = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        # Transpose back: (batch_size, n_heads, seq_len, head_dim)
        # -> (batch_size, seq_len, n_heads, head_dim)
        context = context.transpose(1, 2)

        # Concatenate heads: (batch_size, seq_len, n_heads, head_dim)
        # -> (batch_size, seq_len, d_model)
        context = context.contiguous().view(batch_size, seq_len, self.d_model)

        # Final output projection
        output = self.out_proj(context)

        return output
