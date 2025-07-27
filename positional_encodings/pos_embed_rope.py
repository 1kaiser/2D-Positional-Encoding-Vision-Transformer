import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

class SelfAttentionWithRope(nn.Module):
    """
    Self-attention module with Rotary Positional Embedding (RoPE).
    Equivalent to the original PyTorch SelfAttentionWithRope.
    """
    embed_dim: int
    n_attention_heads: int
    seq_len: int
    # name: str = None # Removed name from __init__ as it's a @nn.compact module

    def setup(self):
        self.head_embed_dim = self.embed_dim // self.n_attention_heads
        if self.head_embed_dim * self.n_attention_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by n_attention_heads")

        # Linear projections for Q, K, V
        self.queries = nn.Dense(features=self.embed_dim, use_bias=False, name='queries')
        self.keys = nn.Dense(features=self.embed_dim, use_bias=False, name='keys')
        self.values = nn.Dense(features=self.embed_dim, use_bias=False, name='values')
        self.out_projection = nn.Dense(features=self.embed_dim, name='out_projection')

        # Precompute RoPE frequencies and rotation factors
        # Frequencies for embed_dim, need embed_dim//2 frequency pairs
        half_head_dim = self.head_embed_dim // 2
        if half_head_dim == 0: # Handle case where head_embed_dim is 1
             freqs = jnp.ones((self.head_embed_dim,))
        else:
            # Use log space for frequencies
            freqs = jnp.exp(jnp.arange(half_head_dim) * -(jnp.log(10000.0) / (half_head_dim - 1))) # (half_head_dim,)
            # Repeat each frequency twice for sin/cos pairs
            freqs = jnp.repeat(freqs, 2) # (half_head_dim * 2,)

            # Ensure frequency dimension matches head_embed_dim
            if freqs.shape[0] < self.head_embed_dim:
                 freqs = jnp.pad(freqs, (0, self.head_embed_dim - freqs.shape[0]), mode='constant', constant_values=1.0)
            elif freqs.shape[0] > self.head_embed_dim:
                 freqs = freqs[:self.head_embed_dim]


        # Compute angles for each position and each frequency
        # positions: (S,) (0 to S-1)
        # freqs: (head_embed_dim,)
        # angles: (S, head_embed_dim) - angle for each position and each dimension
        # The RoPE paper applies frequencies to pairs of dimensions (i, i+1)
        # The angles should be (S, head_embed_dim // 2)
        rope_freq_dim = self.head_embed_dim // 2
        if rope_freq_dim == 0:
             angles = jnp.zeros((self.seq_len, self.head_embed_dim))
        else:
             # Frequencies are for pairs, so we need head_embed_dim // 2 frequencies
             freqs_for_angles = jnp.exp(jnp.arange(rope_freq_dim) * -(jnp.log(10000.0) / (rope_freq_dim - 1))) # (head_embed_dim // 2,)
             angles = jnp.arange(self.seq_len)[:, None] * freqs_for_angles[None, :] # (S, head_embed_dim // 2)

        # Compute cos and sin factors for rotation
        # (S, head_embed_dim // 2) -> (S, head_embed_dim) by repeating sin/cos for each pair
        if rope_freq_dim == 0:
             cos_factors = jnp.ones((self.seq_len, self.head_embed_dim))
             sin_factors = jnp.zeros((self.seq_len, self.head_embed_dim))
        else:
             cos_factors_pairs = jnp.cos(angles) # (S, head_embed_dim // 2)
             sin_factors_pairs = jnp.sin(angles) # (S, head_embed_dim // 2)
             # Repeat each value twice to match the full head dimension
             cos_factors = jnp.repeat(cos_factors_pairs, 2, axis=-1) # (S, head_embed_dim)
             sin_factors = jnp.repeat(sin_factors_pairs, 2, axis=-1) # (S, head_embed_dim)


        # Store as constants (non-learnable state)
        self.cos_factors = cos_factors # (S, HE)
        self.sin_factors = sin_factors # (S, HE)

    # Helper function for applying RoPE rotation
    def _apply_rope(self, x, cos_factors, sin_factors):
        # x: (B, H, S, HE)
        # cos_factors: (S, HE)
        # sin_factors: (S, HE)

        # Split x into two halves for RoPE
        x1, x2 = jnp.split(x, 2, axis=-1) # (B, H, S, HE/2) each

        # Apply rotation formula for pairs: [-x2, x1]
        rotated_x_pairs = jnp.concatenate([-x2, x1], axis=-1) # (B, H, S, HE)

        # Apply the rotation factors element-wise
        x_rotated = x * cos_factors[None, None, :, :] + rotated_x_pairs * sin_factors[None, None, :, :] # (B, H, S, HE)

        return x_rotated


    @nn.compact
    def __call__(self, x):
        b, s, e = x.shape # B, S, E

        # Generate Q, K, V
        # Reshape and transpose to (B, H, S, HE)
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xq = jnp.transpose(xq, (0, 2, 1, 3))                                          # B, H, S, HE
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = jnp.transpose(xk, (0, 2, 1, 3))                                          # B, H, S, HE
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = jnp.transpose(xv, (0, 2, 1, 3))                                          # B, H, S, HE

        # Apply RoPE to queries and keys
        xq_rotated = self._apply_rope(xq, self.cos_factors, self.sin_factors) # (B, H, S, HE)
        xk_rotated = self._apply_rope(xk, self.cos_factors, self.sin_factors) # (B, H, S, HE)

        # Compute Attention presoftmax values using rotated Q and K
        xk_t = jnp.transpose(xk_rotated, (0, 1, 3, 2)) # B, H, HE, S
        x_attention = jnp.matmul(xq_rotated, xk_t)      # B, H, S, S

        x_attention = x_attention / (self.head_embed_dim ** 0.5) # Scale

        x_attention = jax.nn.softmax(x_attention, axis=-1) # Compute Attention Matrix

        x = jnp.matmul(x_attention, xv) # B, H, S, HE

        # Reshape and project output
        x = jnp.transpose(x, (0, 2, 1, 3)) # B, S, H, HE
        x = jnp.reshape(x, (b, s, e))      # B, S, E

        x = self.out_projection(x)         # B, S, E
        return x
