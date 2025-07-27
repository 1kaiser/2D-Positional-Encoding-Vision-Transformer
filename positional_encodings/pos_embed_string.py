# -*- coding: utf-8 -*-
"""
JAX/Flax implementation of Self-Attention with STRING Positional Encoding.
This version is corrected for numerical stability and JAX-native efficiency,
based on the paper arXiv:2502.02562v1.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# B -> Batch Size, H -> Heads, S -> Sequence Length, E -> Embedding Dim, HE -> Head Dim

class StringPositionEmbedding2D(nn.Module):
    """
    STRING: Separable Translationally Invariant Position Encodings.
    This is an optimized JAX implementation.
    """
    seq_len: int
    embed_dim: int  # This is the dimension for a single axis (e.g., HE/2)
    string_type: str = 'cayley'

    def setup(self):
        """Initializes learnable parameters and non-learnable buffers."""
        # --- Learnable Parameters ---
        if self.string_type == 'cayley':
            # Cayley-STRING: Initialize a learnable matrix for the generator
            S_init = nn.initializers.normal(stddev=0.01)
            self.S = self.param('S_cayley', S_init, (self.embed_dim, self.embed_dim))
        elif self.string_type == 'circulant':
            # Circulant-STRING: Initialize the learnable first row
            c_init = nn.initializers.normal(stddev=0.01)
            self.c = self.param('c_circulant', c_init, (self.embed_dim,))
        else:
            raise ValueError(f"Unknown string_type: {self.string_type}")

        # --- Non-Learnable Buffers (Constants) ---
        # Positional indices (0 for CLS, 1 to N for patches)
        self.positions = jnp.arange(self.seq_len, dtype=jnp.float32)

        # Pre-compute base RoPE frequencies for half the dimensions
        self.freqs = 1.0 / (10000 ** (jnp.arange(0, self.embed_dim // 2, dtype=jnp.float32) * 2 / self.embed_dim))

    def _apply_efficient_rope(self, x):
        """Applies RoPE rotation directly to vectors without creating a large matrix."""
        # x shape: (B, H, S, E)
        # Calculate sin/cos factors for each position
        angles = jnp.outer(self.positions, self.freqs)  # (S, E/2)
        cos_vals = jnp.cos(angles)  # (S, E/2)
        sin_vals = jnp.sin(angles)  # (S, E/2)

        # Repeat to match the full embedding dimension
        cos_vals = jnp.repeat(cos_vals, 2, axis=-1)  # (S, E)
        sin_vals = jnp.repeat(sin_vals, 2, axis=-1)  # (S, E)

        # Apply the 2D rotation formula: x_rot = x*cos - permute(x)*sin
        x1, x2 = jnp.split(x, 2, axis=-1)
        x_permuted = jnp.concatenate([-x2, x1], axis=-1)

        # Broadcast across batch and head dimensions
        x_rotated = x * cos_vals[None, None, :, :] + x_permuted * sin_vals[None, None, :, :]
        return x_rotated

    @nn.compact
    def __call__(self, x):
        """Applies STRING positional encoding."""
        # 1. Generate the learnable orthogonal transformation matrix P
        if self.string_type == 'cayley':
            # Make S antisymmetric: (S - S^T)/2
            S_antisym = (self.S - self.S.T) / 2.0
            # Cayley Transform using linear solver for stability, as recommended
            I = jnp.eye(self.embed_dim, dtype=x.dtype)
            P = jnp.linalg.solve(I + S_antisym, I - S_antisym)
        else:  # circulant
            n = len(self.c)
            indices = (jnp.arange(n)[:, None] - jnp.arange(n)[None, :]) % n
            C = self.c[indices]
            P = C - C.T

        # 2. Apply the learnable transformation P to the input
        # (B, H, S, E) @ (E, E) -> (B, H, S, E)
        x_transformed = jnp.matmul(x, P.T)

        # 3. Apply RoPE rotation efficiently to the transformed input
        return self._apply_efficient_rope(x_transformed)


class SelfAttentionWithString(nn.Module):
    """Self-attention module with STRING positional encoding."""
    embed_dim: int
    n_attention_heads: int
    seq_len: int
    string_type: str = 'cayley'

    def setup(self):
        self.head_embed_dim = self.embed_dim // self.n_attention_heads
        if self.head_embed_dim * self.n_attention_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by n_attention_heads")

        self.queries = nn.Dense(features=self.embed_dim, use_bias=False, name='queries')
        self.keys = nn.Dense(features=self.embed_dim, use_bias=False, name='keys')
        self.values = nn.Dense(features=self.embed_dim, use_bias=False, name='values')
        self.out_projection = nn.Dense(features=self.embed_dim, name='out_projection')

        # Split head_embed_dim for x and y axes, handling odd dimensions
        self.dim_split_x = self.head_embed_dim // 2 + (self.head_embed_dim % 2)
        self.dim_split_y = self.head_embed_dim // 2

        self.string_x = StringPositionEmbedding2D(seq_len=self.seq_len, embed_dim=self.dim_split_x, string_type=self.string_type, name='string_x')
        self.string_y = StringPositionEmbedding2D(seq_len=self.seq_len, embed_dim=self.dim_split_y, string_type=self.string_type, name='string_y')

    @nn.compact
    def __call__(self, x):
        b, s, e = x.shape

        # Generate Q, K, V and reshape to (B, H, S, HE)
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xq = jnp.transpose(xq, (0, 2, 1, 3))
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = jnp.transpose(xk, (0, 2, 1, 3))
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = jnp.transpose(xv, (0, 2, 1, 3))

        # Split queries and keys for x and y axes
        xq_x, xq_y = jnp.split(xq, [self.dim_split_x], axis=-1)
        xk_x, xk_y = jnp.split(xk, [self.dim_split_x], axis=-1)

        # Apply STRING positional encoding to each axis
        xq_x_str = self.string_x(xq_x)
        xq_y_str = self.string_y(xq_y)
        xk_x_str = self.string_x(xk_x)
        xk_y_str = self.string_y(xk_y)

        # Concatenate the results
        xq = jnp.concatenate([xq_x_str, xq_y_str], axis=-1)
        xk = jnp.concatenate([xk_x_str, xk_y_str], axis=-1)

        # Standard attention computation
        xk_t = jnp.transpose(xk, (0, 1, 3, 2))
        x_attention = jnp.matmul(xq, xk_t) / np.sqrt(self.head_embed_dim)
        x_attention = nn.softmax(x_attention, axis=-1)

        x = jnp.matmul(x_attention, xv)

        # Reshape and project output
        x = jnp.transpose(x, (0, 2, 1, 3))
        x = x.reshape(b, s, e)
        x = self.out_projection(x)

        return x
