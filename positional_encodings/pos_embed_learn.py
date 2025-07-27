import flax.linen as nn
import jax.numpy as jnp
import numpy as np # Use numpy for shape calculations if needed

class EmbedLayerWithLearn(nn.Module):
    """
    Embedding layer for Vision Transformer with learnable positional encoding.
    Equivalent to the original PyTorch EmbedLayerWithLearn.
    """
    n_channels: int
    embed_dim: int
    image_size: int
    patch_size: int
    # dropout_rate: float = 0.0 # Removed from __init__

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
        n_patches = h * w
        x = jnp.reshape(x, (b, n_patches, c)) # Flatten spatial dimensions

        # Classification Token (learnable parameter)
        cls_token = self.param('cls_token',
                               nn.initializers.zeros,
                               (1, 1, self.embed_dim)) # (1, 1, E)
        cls_token = jnp.tile(cls_token, [b, 1, 1])    # Repeat for batch size (B, 1, E)

        # Concatenate Classification Token and Patch Embeddings
        x = jnp.concatenate([cls_token, x], axis=1) # (B, N+1, E)
        seq_len = x.shape[1] # S = N + 1

        # Learnable Positional Embedding (learnable parameter)
        # Corrected 'std' to 'stddev' for Flax initializer
        pos_embed = self.param('pos_embed',
                               nn.initializers.normal(stddev=0.02), # Use similar initialization
                               (1, seq_len, self.embed_dim)) # (1, S, E)
        # Add positional embedding
        x = x + pos_embed # (B, S, E)

        # Dropout - use the dropout_rate passed to __call__
        x = nn.Dropout(rate=dropout_rate, deterministic=not train)(x)

        return x
