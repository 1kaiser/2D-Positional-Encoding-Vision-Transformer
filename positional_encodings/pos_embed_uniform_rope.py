# filename: positional_encodings/pos_embed_uniform_rope.py

# @misc{xiong2025ndrope
#     author = {Jerry Xiong},
#     title = {On N-dimensional rotary positional embeddings},
#     year = {2025},
#     url = {https://jerryxio.ng/posts/nd-rope/}
# }

import flax.linen as nn
import jax.numpy as jnp
import jax
from jax.scipy.special import erfinv
import numpy as np
import math
from typing import Tuple

# --- Helper functions for UniformRoPENdFlax ---

def _phi_jax(m: int) -> float:
    """Helper function to find the base for the golden ratio spiral."""
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x

def uniform_directions_jax(n: int, d: int) -> jnp.ndarray:
    """Generates quasi-random uniformly distributed direction vectors in d dimensions."""
    g = _phi_jax(d)
    alpha = (1.0 / g) ** jnp.arange(1, d + 1, dtype=jnp.float64)
    i = jnp.arange(1, n + 1, dtype=jnp.float64)[:, jnp.newaxis]
    z = jnp.fmod(i * alpha, 1.0)
    # Use inverse error function to map uniform distribution to a normal distribution
    directions = erfinv(2.0 * z - 1.0)
    # Normalize to get unit vectors
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    return directions.astype(jnp.float32)

# --- Uniform RoPE Module ---

class UniformRoPENdFlax(nn.Module):
    """
    JAX/Flax implementation of Uniform N-dimensional RoPE (dynamic).
    Rotates token embeddings based on their N-dimensional positions.
    """
    pos_dim: int
    n_heads: int
    head_dim: int
    min_freq: float = 0.1
    max_freq: float = 1000.0
    p_zero_freqs: float = 0.0

    def setup(self):
        assert self.head_dim % 2 == 0, "head_dim must be even."
        n_freqs = self.head_dim // 2
        n_zero_freqs = round(self.p_zero_freqs * n_freqs)
        
        # Create frequency magnitudes (log-spaced)
        omega_F = jnp.concatenate([
            jnp.zeros(n_zero_freqs),
            self.min_freq * (self.max_freq / self.min_freq) ** jnp.linspace(0, 1, n_freqs - n_zero_freqs)
        ])
        
        # Generate N-dimensional direction vectors
        directions_hFP = uniform_directions_jax(self.n_heads * n_freqs, self.pos_dim).reshape(
            self.n_heads, n_freqs, self.pos_dim
        )
        # Store frequency vectors as a non-trainable attribute
        self.freqs_hFP = directions_hFP * omega_F.reshape(1, n_freqs, 1)

    def __call__(self, input_NLhd: jnp.ndarray, pos_NLP: jnp.ndarray) -> jnp.ndarray:
        """
        Applies dynamically computed rotations.
        Shape key: N:batch, L:sequence_length, P:pos_dim, h:heads, d:dims, F:freqs
        """
        x_NLhF, y_NLhF = jnp.split(input_NLhd, 2, axis=-1)
        
        # Expand positions for broadcasting and compute inner product with frequencies
        pos_expanded = pos_NLP[:, :, jnp.newaxis, jnp.newaxis, :]
        theta_NLhF = (self.freqs_hFP * pos_expanded).sum(axis=-1)
        
        cos_NLhF, sin_NLhF = jnp.cos(theta_NLhF), jnp.sin(theta_NLhF)
        
        # Apply rotation
        x_out_NLhF = x_NLhF * cos_NLhF - y_NLhF * sin_NLhF
        y_out_NLhF = x_NLhF * sin_NLhF + y_NLhF * cos_NLhF
        
        return jnp.concatenate([x_out_NLhF, y_out_NLhF], axis=-1)

# --- Self-Attention Module with Uniform RoPE ---

class SelfAttentionWithUniformRope(nn.Module):
    """Self-attention module with Uniform RoPE positional encoding."""
    embed_dim: int
    n_attention_heads: int
    seq_len: int

    def setup(self):
        self.head_embed_dim = self.embed_dim // self.n_attention_heads
        if self.head_embed_dim * self.n_attention_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by n_attention_heads")

        self.queries = nn.Dense(features=self.embed_dim, use_bias=False, name='queries')
        self.keys = nn.Dense(features=self.embed_dim, use_bias=False, name='keys')
        self.values = nn.Dense(features=self.embed_dim, use_bias=False, name='values')
        self.out_projection = nn.Dense(features=self.embed_dim, name='out_projection')
        
        # Instantiate the Uniform RoPE encoder for 2D positions
        self.uniform_rope_encoder = UniformRoPENdFlax(
            pos_dim=2,
            n_heads=self.n_attention_heads,
            head_dim=self.head_embed_dim,
            name='uniform_rope_encoder'
        )

    def _get_2d_coordinates(self):
        """Generate normalized 2D patch coordinates in the range [-1, 1]."""
        n_patches_per_dim = int(np.sqrt(self.seq_len - 1))
        
        y_coords, x_coords = jnp.meshgrid(
            jnp.arange(n_patches_per_dim),
            jnp.arange(n_patches_per_dim),
            indexing='ij'
        )
        
        # Normalize coordinates to be in [-1, 1] for stability
        # Handle the case of a single patch to avoid division by zero
        if n_patches_per_dim > 1:
            x_patch_coords = (x_coords.flatten().astype(jnp.float32) / (n_patches_per_dim - 1)) * 2 - 1
            y_patch_coords = (y_coords.flatten().astype(jnp.float32) / (n_patches_per_dim - 1)) * 2 - 1
        else:
            x_patch_coords = jnp.array([0.0])
            y_patch_coords = jnp.array([0.0])

        # Add CLS token coordinates (0, 0), which is the center of the normalized grid
        cls_coords = jnp.array([[0.0, 0.0]])
        patch_coords = jnp.stack([x_patch_coords, y_patch_coords], axis=1)
        
        # Full coordinates with CLS token first
        coords_full = jnp.concatenate([cls_coords, patch_coords], axis=0)
        return coords_full

    @nn.compact
    def __call__(self, x, depth_map=None): # Added depth_map for API consistency
        b, s, e = x.shape

        # Generate Q, K, V and reshape to (B, S, H, HE)
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        
        # Get 2D patch coordinates
        coords = self._get_2d_coordinates() # Shape (S, 2)
        
        # Apply Uniform RoPE to queries and keys
        # Add a batch dimension to coords for broadcasting
        coords_batch = jnp.broadcast_to(coords, (b, s, 2))
        xq = self.uniform_rope_encoder(xq, coords_batch)
        xk = self.uniform_rope_encoder(xk, coords_batch)
        
        # Transpose for attention calculation: (B, H, S, HE)
        xq = jnp.transpose(xq, (0, 2, 1, 3))
        xk = jnp.transpose(xk, (0, 2, 1, 3))
        xv = jnp.transpose(xv, (0, 2, 1, 3))

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
