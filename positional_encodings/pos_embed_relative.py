# -*- coding: utf-8 -*-
"""
JAX/Flax implementation of Self-Attention with Relative Positional Encoding.
This version is corrected to be compatible with JIT compilation.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

class SelfAttentionWithRelative(nn.Module):
    """
    Self-attention module with learned relative positional encoding bias.
    This version is corrected to be JIT-compatible by using jnp.where
    instead of direct boolean array indexing.
    """
    embed_dim: int
    n_attention_heads: int
    seq_len: int
    max_relative_dist: int = 2 # Referred to as k in the paper

    def setup(self):
        """
        Initializes layers and pre-computes the relative position index matrix
        in a JIT-friendly manner.
        """
        self.head_embed_dim = self.embed_dim // self.n_attention_heads
        if self.head_embed_dim * self.n_attention_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by n_attention_heads")

        # Linear projections for Q, K, V
        self.queries = nn.Dense(features=self.embed_dim, use_bias=False, name='queries')
        self.keys = nn.Dense(features=self.embed_dim, use_bias=False, name='keys')
        self.values = nn.Dense(features=self.embed_dim, use_bias=False, name='values')
        self.out_projection = nn.Dense(features=self.embed_dim, name='out_projection')

        n_patches = self.seq_len - 1
        n_patches_per_dim = int(np.sqrt(n_patches))

        # Total number of entries in the learnable bias table
        num_relative_bias_entries = (2 * self.max_relative_dist - 1)**2

        # Define specific indices for CLS token interactions in the bias table
        cls_cls_index = num_relative_bias_entries
        cls_patch_index = num_relative_bias_entries + 1
        patch_cls_index = num_relative_bias_entries + 2

        # The relative_bias_table stores biases for all relative positions + CLS interactions
        self.relative_bias_table = self.param(
            'relative_bias_table',
            nn.initializers.zeros,
            (num_relative_bias_entries + 3,) # Patch-patch + 3 CLS cases
        )

        # --- JAX-friendly index calculation ---
        # Create 1D indices for the entire sequence (0 for CLS, 1 to N for patches)
        query_indices = jnp.arange(self.seq_len)
        key_indices = jnp.arange(self.seq_len)

        # Calculate 2D coordinates for all sequence positions.
        # We subtract 1 to map sequence indices [1, N] to patch coordinates [0, N-1].
        # The CLS token (index 0) will have coordinates (-1, -1), which is fine as masks will handle it.
        q_coords_y = (query_indices - 1) // n_patches_per_dim
        q_coords_x = (query_indices - 1) % n_patches_per_dim
        k_coords_y = (key_indices - 1) // n_patches_per_dim
        k_coords_x = (key_indices - 1) % n_patches_per_dim

        # Calculate relative coordinates for the entire grid (S, S) using broadcasting
        relative_coords_y = q_coords_y[:, None] - k_coords_y[None, :]
        relative_coords_x = q_coords_x[:, None] - k_coords_x[None, :]

        # Clamp relative coordinates to the range [-k+1, k-1]
        relative_coords_y = jnp.clip(relative_coords_y, -self.max_relative_dist + 1, self.max_relative_dist - 1)
        relative_coords_x = jnp.clip(relative_coords_x, -self.max_relative_dist + 1, self.max_relative_dist - 1)

        # Map the clamped 2D relative coordinates to a 1D index for the bias table
        patch_indices = (relative_coords_y + self.max_relative_dist - 1) * (2 * self.max_relative_dist - 1) + \
                        (relative_coords_x + self.max_relative_dist - 1)

        # --- Use jnp.where to handle CLS token vs. Patches ---
        # Create boolean masks to identify CLS token positions
        q_is_cls = (query_indices == 0)
        k_is_cls = (key_indices == 0)

        # Build the final index matrix by applying conditions. This avoids direct boolean indexing.
        # 1. Start with the calculated patch-to-patch indices.
        final_indices = patch_indices
        # 2. Where key is CLS token, overwrite with the patch_cls_index.
        final_indices = jnp.where(k_is_cls[None, :], patch_cls_index, final_indices)
        # 3. Where query is CLS token, overwrite with the cls_patch_index.
        final_indices = jnp.where(q_is_cls[:, None], cls_patch_index, final_indices)
        # 4. Where both are CLS tokens (top-left corner), overwrite with cls_cls_index.
        final_indices = jnp.where(q_is_cls[:, None] & k_is_cls[None, :], cls_cls_index, final_indices)

        # Store the final, non-learnable index matrix (S, S)
        self.relative_position_indices = final_indices


    @nn.compact
    def __call__(self, x):
        b, s, e = x.shape # B, S, E

        # Generate Q, K, V
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xq = jnp.transpose(xq, (0, 2, 1, 3)) # (B, H, S, HE)
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = jnp.transpose(xk, (0, 2, 1, 3)) # (B, H, S, HE)
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = jnp.transpose(xv, (0, 2, 1, 3)) # (B, H, S, HE)

        # Compute Attention presoftmax values
        xk_t = jnp.transpose(xk, (0, 1, 3, 2)) # (B, H, HE, S)
        x_attention = jnp.matmul(xq, xk_t)      # (B, H, S, S)

        # Scale by head dimension
        x_attention = x_attention / (self.head_embed_dim ** 0.5)

        # Add relative positional bias
        # Retrieve biases using the precomputed index matrix. This is JIT-friendly.
        relative_bias = jnp.take(self.relative_bias_table, self.relative_position_indices, axis=0) # (S, S)

        # Add the bias to the attention scores, broadcasting across batch and heads
        x_attention = x_attention + relative_bias[None, None, :, :] # (B, H, S, S)

        # Compute Attention Matrix
        x_attention = jax.nn.softmax(x_attention, axis=-1)

        # Apply attention to values
        x = jnp.matmul(x_attention, xv) # (B, H, S, HE)

        # Reshape and project output
        x = jnp.transpose(x, (0, 2, 1, 3)) # (B, S, H, HE)
        x = jnp.reshape(x, (b, s, e))      # (B, S, E)
        x = self.out_projection(x)         # (B, S, E)

        return x
