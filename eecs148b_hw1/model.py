"""
Transformer language model components — implemented from scratch.

Sections 3.1 – 3.3 of the assignment:
  Linear, Embedding, LayerNorm, FFN, SinusoidalPositionalEncoding,
  scaled_dot_product_attention, softmax, MultiHeadSelfAttention,
  TransformerBlock, TransformerLM
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 3.1.2  Linear
# ---------------------------------------------------------------------------

class Linear(nn.Module):
    """Linear transformation y = W x  (no bias), matching nn.Linear interface."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store W (out × in) so the forward pass is x @ W.T
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


# ---------------------------------------------------------------------------
# 3.1.3  Embedding
# ---------------------------------------------------------------------------

class Embedding(nn.Module):
    """Token embedding lookup, matching nn.Embedding interface."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


# ---------------------------------------------------------------------------
# 3.2.1  LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Standard LayerNorm with learnable affine transform."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        result = x_norm * self.weight + self.bias
        return result.to(in_dtype)


# ---------------------------------------------------------------------------
# 3.2.2  Position-Wise Feed-Forward Network
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Two-layer ReLU feed-forward network: FFN(x) = W2(ReLU(W1 x))."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.fc2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU: max(0, x) via multiplication by boolean mask
        h = self.fc1(x)
        h = h * (h > 0)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# 3.2.3  Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional embeddings (Vaswani et al., 2017)."""

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # Precompute (max_seq_len, d_model) table
        pe = torch.zeros(max_seq_len, d_model)
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)   # (T, 1)
        # i indices for even dimensions: 0, 1, 2, ...  → dimension 2i and 2i+1
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        if dtype is not None:
            pe = pe.to(dtype)
        if device is not None:
            pe = pe.to(device)

        # Non-persistent: not saved in state_dict, recomputed on load
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions: (..., seq_len) → output: (..., seq_len, d_model)
        return self.pe[token_positions]


# ---------------------------------------------------------------------------
# 3.2.4  Softmax + Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax along `dim`."""
    x_max = x.amax(dim=dim, keepdim=True)
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    Args:
        Q: (..., n_queries, d_k)
        K: (..., n_keys,   d_k)
        V: (..., n_keys,   d_v)
        mask: (n_queries, n_keys) bool tensor.
              True  → attend, False → do not attend.

    Returns:
        (..., n_queries, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (..., n_q, n_k)

    if mask is not None:
        # Where mask is False, set score to -inf so softmax → 0
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)
    return attn @ V


# ---------------------------------------------------------------------------
# 3.2.5  Causal Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Cache causal mask up to a large size; expanded lazily per actual T
        self._causal_cache_size = 0
        self.register_buffer("_causal_mask", torch.zeros(0, 0, dtype=torch.bool), persistent=False)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if T > self._causal_cache_size:
            self._causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
            self._causal_cache_size = T
        return self._causal_mask[:T, :T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h, d_k = self.num_heads, self.d_k

        Q = self.q_proj(x).view(B, T, h, d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, h, d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, h, d_k).transpose(1, 2)

        causal_mask = self._get_causal_mask(T, x.device)
        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        out = out.transpose(1, 2).contiguous().view(B, T, h * d_k)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# 3.3  Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (LayerNorm → sublayer → residual)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_layernorm: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm
        self.ln1 = LayerNorm(d_model, device=device, dtype=dtype) if use_layernorm else nn.Identity()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = LayerNorm(d_model, device=device, dtype=dtype) if use_layernorm else nn.Identity()
        self.ffn = FFN(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# 3.3  Full Transformer Language Model
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """Transformer language model with sinusoidal positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        use_layernorm: bool = True,
        use_pos_emb: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.use_pos_emb = use_pos_emb

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.pos_embeddings = (
            SinusoidalPositionalEncoding(d_model, context_length, device=device, dtype=dtype)
            if use_pos_emb else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, use_layernorm=use_layernorm, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = LayerNorm(d_model, device=device, dtype=dtype) if use_layernorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) int tensor

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = token_ids.shape

        x = self.token_embeddings(token_ids)
        if self.use_pos_emb:
            positions = torch.arange(T, device=token_ids.device)
            x = x + self.pos_embeddings(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# 4.1  Cross-Entropy Loss
# ---------------------------------------------------------------------------

def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable cross-entropy loss averaged over all examples.

    Args:
        logits:  (..., vocab_size) — unnormalised log-probabilities
        targets: (...,)            — integer class indices

    Returns:
        Scalar mean loss.
    """
    # Shift logits for numerical stability: subtract per-example max
    logits_max = logits.amax(dim=-1, keepdim=True)
    shifted = logits - logits_max

    # log-sum-exp of the shifted logits
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))

    # Gather the log-probability of the correct class
    # targets shape: (...) → need to index the last dim of shifted
    log_prob_correct = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    loss = -log_prob_correct + log_sum_exp
    return loss.mean()
