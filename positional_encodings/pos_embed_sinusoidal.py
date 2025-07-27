import flax.linen as nn
import jax.numpy as jnp
import numpy as np

class EmbedLayerWithSinusoidal(nn.Module):
    """
    Embedding layer for Vision Transformer with fixed sinusoidal positional encoding.
    Equivalent to the original PyTorch EmbedLayerWithSinusoidal.
    """
    n_channels: int
    embed_dim: int
    image_size: int
    patch_size: int

    def setup(self):
        # Precompute sinusoidal positional embeddings
        n_patches_per_dim = self.image_size // self.patch_size
        n_patches = n_patches_per_dim * n_patches_per_dim
        seq_len = n_patches + 1 # Including CLS token

        # Generate 2D positions for patches (0 to n_patches_per_dim-1)
        y_grid, x_grid = jnp.meshgrid(jnp.arange(n_patches_per_dim), jnp.arange(n_patches_per_dim), indexing='ij')
        x_pos_patch = x_grid.flatten() # (N,)
        y_pos_patch = y_grid.flatten() # (N,)

        # Add position 0 for CLS token
        x_pos = jnp.concatenate([jnp.array([0]), x_pos_patch]) # (S,)
        y_pos = jnp.concatenate([jnp.array([0]), y_pos_patch]) # (S,)

        # Compute frequencies for sinusoidal embeddings
        # Split embed_dim into two halves for x and y encoding
        half_embed_dim = self.embed_dim // 2
        # Frequencies for each half (use half_embed_dim // 2 frequency pairs for each)
        freq_dim = half_embed_dim // 2
        if freq_dim == 0: # Handle case where half_embed_dim is 1
             freqs = jnp.ones((half_embed_dim,))
        else:
            # Use log space for frequencies
            freqs = jnp.exp(jnp.arange(freq_dim) * -(jnp.log(10000.0) / (freq_dim - 1))) # (freq_dim,)
            # Repeat each frequency twice for sin/cos pairs
            freqs = jnp.repeat(freqs, 2) # (freq_dim * 2,)

            # Ensure frequencies match the half_embed_dim
            if freqs.shape[0] < half_embed_dim:
                 freqs = jnp.pad(freqs, (0, half_embed_dim - freqs.shape[0]), mode='constant', constant_values=1.0)
            elif freqs.shape[0] > half_embed_dim:
                 freqs = freqs[:half_embed_dim]

        # Compute embeddings for x and y separately
        # (S, 1) * (1, half_embed_dim) -> (S, half_embed_dim)
        x_embed = x_pos[:, None] * freqs[None, :] # (S, half_embed_dim)
        y_embed = y_pos[:, None] * freqs[None, :] # (S, half_embed_dim)

        # Apply sin and cos for x and y
        x_sin = jnp.sin(x_embed) # (S, half_embed_dim)
        x_cos = jnp.cos(x_embed) # (S, half_embed_dim)
        y_sin = jnp.sin(y_embed) # (S, half_embed_dim)
        y_cos = jnp.cos(y_embed) # (S, half_embed_dim)

        # Concatenate x and y embeddings
        # [x_sin, x_cos] form the x-encoding (S, half_embed_dim)
        # [y_sin, y_cos] form the y-encoding (S, half_embed_dim)
        # Concatenate x-encoding and y-encoding
        # Result shape (S, embed_dim)
        pos_embed = jnp.concatenate([x_sin, x_cos, y_sin, y_cos], axis=-1) # (S, 2 * half_embed_dim) which is (S, embed_dim)

        # Handle odd embed_dim: If embed_dim is odd, half_embed_dim = (embed_dim-1)//2.
        # 2 * half_embed_dim = embed_dim - 1. We need to pad one dimension.
        if self.embed_dim % 2 != 0:
             # The standard approach is to encode x and y using embed_dim//2 dimensions each,
             # and the last dimension is either dropped or handled specially.
             # Let's adjust the split if embed_dim is odd.
             # Let first half be ceil(embed_dim/2), second half floor(embed_dim/2).
             half_embed_dim_x = self.embed_dim // 2 + (self.embed_dim % 2)
             half_embed_dim_y = self.embed_dim // 2

             # Recalculate frequencies based on the new half dimensions
             freq_dim_x = half_embed_dim_x // 2
             freq_dim_y = half_embed_dim_y // 2

             if freq_dim_x == 0: freqs_x = jnp.ones((half_embed_dim_x,))
             else:
                 freqs_x = jnp.exp(jnp.arange(freq_dim_x) * -(jnp.log(10000.0) / (freq_dim_x - 1)))
                 freqs_x = jnp.repeat(freqs_x, 2)
                 if freqs_x.shape[0] < half_embed_dim_x: freqs_x = jnp.pad(freqs_x, (0, half_embed_dim_x - freqs_x.shape[0]), mode='constant', constant_values=1.0)
                 elif freqs_x.shape[0] > half_embed_dim_x: freqs_x = freqs_x[:half_embed_dim_x]

             if freq_dim_y == 0: freqs_y = jnp.ones((half_embed_dim_y,))
             else:
                 freqs_y = jnp.exp(jnp.arange(freq_dim_y) * -(jnp.log(10000.0) / (freq_dim_y - 1)))
                 freqs_y = jnp.repeat(freqs_y, 2)
                 if freqs_y.shape[0] < half_embed_dim_y: freqs_y = jnp.pad(freqs_y, (0, half_embed_dim_y - freqs_y.shape[0]), mode='constant', constant_values=1.0)
                 elif freqs_y.shape[0] > half_embed_dim_y: freqs_y = freqs_y[:half_embed_dim_y]

             # Compute embeddings for x and y separately with adjusted dimensions
             x_embed = x_pos[:, None] * freqs_x[None, :] # (S, half_embed_dim_x)
             y_embed = y_pos[:, None] * freqs_y[None, :] # (S, half_embed_dim_y)

             x_sin = jnp.sin(x_embed) # (S, half_embed_dim_x)
             x_cos = jnp.cos(x_embed) # (S, half_embed_dim_x)
             y_sin = jnp.sin(y_embed) # (S, half_embed_dim_y)
             y_cos = jnp.cos(y_embed) # (S, half_embed_dim_y)

             # Concatenate x and y embeddings
             pos_embed = jnp.concatenate([x_sin, x_cos, y_sin, y_cos], axis=-1) # (S, 2*half_embed_dim_x + 2*half_embed_dim_y)
             # This still results in (S, 2 * embed_dim) or (S, 2*embed_dim + 2) if embed_dim is odd.

             # Let's use the standard 2D sinusoidal PE formulation:
             # Split the embed_dim into 4 parts (or as close as possible) for sin/cos of x and sin/cos of y.
             # Or split into 2 halves, first half for x, second for y. Within each half, use sin/cos pairs.
             # Let's split embed_dim into two halves (x and y), then apply sin/cos to each half.
             # The frequencies should be calculated for embed_dim // 2.
             # The angles for x are pos_x * freqs (S, embed_dim//2)
             # The angles for y are pos_y * freqs (S, embed_dim//2)
             # x_sin, x_cos, y_sin, y_cos are all (S, embed_dim//2)
             # Concatenate [x_sin, x_cos, y_sin, y_cos] leads to (S, 2 * embed_dim). This is the source of the error.

             # The correct concatenation for (S, embed_dim) is to interleave the sin/cos results:
             # pos_embed[pos, 2i] = sin(pos_x * freqs[i])
             # pos_embed[pos, 2i+1] = cos(pos_x * freqs[i])
             # for i from 0 to embed_dim//4 - 1.
             # pos_embed[pos, embed_dim//2 + 2i] = sin(pos_y * freqs[i])
             # pos_embed[pos, embed_dim//2 + 2i+1] = cos(pos_y * freqs[i])
             # for i from 0 to embed_dim//4 - 1.

             # Let's recalculate freqs for embed_dim // 4
             freq_dim = self.embed_dim // 4
             if freq_dim == 0: freqs = jnp.ones((self.embed_dim // 2,)) # Fallback if embed_dim < 4
             else:
                 freqs = jnp.exp(jnp.arange(freq_dim) * -(jnp.log(10000.0) / (freq_dim - 1))) # (freq_dim,)

             # Angles for x and y
             x_angles = x_pos[:, None] * freqs[None, :] # (S, freq_dim)
             y_angles = y_pos[:, None] * freqs[None, :] # (S, freq_dim)

             # Sin/Cos for x and y angles
             x_sin = jnp.sin(x_angles) # (S, freq_dim)
             x_cos = jnp.cos(x_angles) # (S, freq_dim)
             y_sin = jnp.sin(y_angles) # (S, freq_dim)
             y_cos = jnp.cos(y_angles) # (S, freq_dim)

             # Interleave/Concatenate to get (S, embed_dim)
             # [x_sin, x_cos, y_sin, y_cos] concatenated gives (S, 4 * freq_dim) which is (S, embed_dim)
             pos_embed = jnp.concatenate([x_sin, x_cos, y_sin, y_cos], axis=-1) # (S, embed_dim)

             # Handle cases where embed_dim is not divisible by 4
             if pos_embed.shape[-1] < self.embed_dim:
                  # Pad with zeros if the dimension is less than embed_dim
                  pos_embed = jnp.pad(pos_embed, ((0, 0), (0, self.embed_dim - pos_embed.shape[-1])), mode='constant')
             elif pos_embed.shape[-1] > self.embed_dim:
                  # Truncate if the dimension is more than embed_dim (shouldn't happen with this logic)
                  pos_embed = pos_embed[:, :self.embed_dim]

        else: # embed_dim is even and >= 2
             # Standard 2D sinusoidal PE
             half_embed_dim = self.embed_dim // 2
             freq_dim = half_embed_dim // 2
             if freq_dim == 0: freqs = jnp.ones((half_embed_dim,))
             else:
                 freqs = jnp.exp(jnp.arange(freq_dim) * -(jnp.log(10000.0) / (freq_dim - 1))) # (freq_dim,)

             x_angles = x_pos[:, None] * freqs[None, :] # (S, freq_dim)
             y_angles = y_pos[:, None] * freqs[None, :] # (S, freq_dim)

             x_sin = jnp.sin(x_angles) # (S, freq_dim)
             x_cos = jnp.cos(x_angles) # (S, freq_dim)
             y_sin = jnp.sin(y_angles) # (S, freq_dim)
             y_cos = jnp.cos(y_angles) # (S, freq_dim)

             # Concatenate [x_sin, x_cos, y_sin, y_cos]
             pos_embed = jnp.concatenate([x_sin, x_cos, y_sin, y_cos], axis=-1) # (S, 4 * freq_dim) which is (S, embed_dim)


        # Store as a constant array
        self.pos_embed = pos_embed # (S, E)


    @nn.compact
    def __call__(self, x, train: bool, dropout_rate: float = 0.0):
        b, h_img, w_img, c_img = x.shape
        # Image to Patch Embedding
        # Input: (B, H, W, C) -> Output: (B, N, E) where N = (H/P)*(W/P)
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='patch_embedding'
        )(x)
        b, h, w, c = x.shape
        x = jnp.reshape(x, (b, h * w, c)) # Flatten spatial dimensions

        # Classification Token (learnable parameter)
        cls_token = self.param('cls_token',
                               nn.initializers.zeros,
                               (1, 1, self.embed_dim)) # (1, 1, E)
        cls_token = jnp.tile(cls_token, [b, 1, 1])    # Repeat for batch size (B, 1, E)

        # Concatenate Classification Token and Patch Embeddings
        x = jnp.concatenate([cls_token, x], axis=1) # (B, N+1, E)

        # Add precomputed sinusoidal positional embedding
        x = x + self.pos_embed # (B, S, E)

        # Dropout - use the dropout_rate passed to __call__
        x = nn.Dropout(rate=dropout_rate, deterministic=not train)(x)

        return x
