import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# Import converted positional encoding modules
from positional_encodings.pos_embed_none import EmbedLayerWithNone
from positional_encodings.pos_embed_learn import EmbedLayerWithLearn
from positional_encodings.pos_embed_sinusoidal import EmbedLayerWithSinusoidal
from positional_encodings.pos_embed_relative import SelfAttentionWithRelative
from positional_encodings.pos_embed_rope import SelfAttentionWithRope
from positional_encodings.pos_embed_string import SelfAttentionWithString


# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# N -> Number of Patches = IH/P * IW/P
# S -> Sequence Length   = IH/P * IW/P + 1 or N + 1 (extra 1 is of Classification Token)
# Q -> Query Sequence length (equal to S for self-attention)
# K -> Key Sequence length   (equal to S for self-attention)
# V -> Value Sequence length (equal to S for self-attention)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class OriginalSelfAttention(nn.Module):
    """
    Original Self-attention module without positional encoding modifications.
    Equivalent to the original PyTorch OriginalSelfAttention.
    """
    embed_dim: int
    n_attention_heads: int

    def setup(self):
        self.head_embed_dim = self.embed_dim // self.n_attention_heads
        if self.head_embed_dim * self.n_attention_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by n_attention_heads")

        # Linear projections for Q, K, V
        self.queries = nn.Dense(features=self.embed_dim, use_bias=False, name='queries')
        self.keys = nn.Dense(features=self.embed_dim, use_bias=False, name='keys')
        self.values = nn.Dense(features=self.embed_dim, use_bias=False, name='values')
        self.out_projection = nn.Dense(features=self.embed_dim, name='out_projection')

    @nn.compact
    def __call__(self, x):
        b, s, e = x.shape # B, S, E

        # Generate Q, K, V
        # Reshape and transpose to (B, H, S, HE)
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xq = jnp.transpose(xq, (0, 2, 1, 3)) # B, H, S, HE
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = jnp.transpose(xk, (0, 2, 1, 3)) # B, H, S, HE
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = jnp.transpose(xv, (0, 2, 1, 3)) # B, H, S, HE


        # Compute Attention presoftmax values
        xk_t = jnp.transpose(xk, (0, 1, 3, 2)) # B, H, HE, S
        x_attention = jnp.matmul(xq, xk_t)      # B, H, S, S

        x_attention = x_attention / (self.head_embed_dim ** 0.5) # Scale

        x_attention = jax.nn.softmax(x_attention, axis=-1) # Compute Attention Matrix

        x = jnp.matmul(x_attention, xv) # B, H, S, HE

        # Reshape and project output
        x = jnp.transpose(x, (0, 2, 1, 3)) # B, S, H, HE
        x = jnp.reshape(x, (b, s, e))      # B, S, E

        x = self.out_projection(x)         # B, S, E
        return x


class Encoder(nn.Module):
    """
    Transformer Encoder block.
    Equivalent to the original PyTorch Encoder.
    """
    embed_dim: int
    n_attention_heads: int
    forward_mul: int
    seq_len: int
    dropout_rate: float = 0.0
    pos_embed: str = 'learn'
    max_relative_distance: int = 2
    string_type: str = 'cayley'

    @nn.compact
    def __call__(self, x, train: bool):
        # Layer Normalization
        norm1_out = nn.LayerNorm(name='norm1')(x)

        # Attention
        if self.pos_embed == 'relative':
            attention_out = SelfAttentionWithRelative(
                embed_dim=self.embed_dim,
                n_attention_heads=self.n_attention_heads,
                seq_len=self.seq_len,
                max_relative_dist=self.max_relative_distance,
                name='attention'
            )(norm1_out)
        elif self.pos_embed == 'rope':
            attention_out = SelfAttentionWithRope(
                embed_dim=self.embed_dim,
                n_attention_heads=self.n_attention_heads,
                seq_len=self.seq_len,
                name='attention'
            )(norm1_out)
        elif self.pos_embed == 'string':
             attention_out = SelfAttentionWithString(
                 embed_dim=self.embed_dim,
                 n_attention_heads=self.n_attention_heads,
                 seq_len=self.seq_len,
                 string_type=self.string_type,
                 name='attention'
             )(norm1_out)
        else: # 'none', 'learn', 'sinusoidal' use OriginalSelfAttention
            attention_out = OriginalSelfAttention(
                embed_dim=self.embed_dim,
                n_attention_heads=self.n_attention_heads,
                name='attention'
            )(norm1_out)

        # Dropout and Skip connection
        x = x + nn.Dropout(rate=self.dropout_rate, deterministic=not train)(attention_out)

        # Layer Normalization
        norm2_out = nn.LayerNorm(name='norm2')(x)

        # Feedforward
        fc1_out = nn.Dense(features=self.embed_dim * self.forward_mul, name='fc1')(norm2_out)
        activation_out = jax.nn.gelu(fc1_out) # GELU activation
        fc2_out = nn.Dense(features=self.embed_dim, name='fc2')(activation_out)

        # Dropout and Skip connection
        x = x + nn.Dropout(rate=self.dropout_rate, deterministic=not train)(fc2_out)

        return x


class Classifier(nn.Module):
    """
    Classifier head for Vision Transformer, taking the CLS token embedding.
    Equivalent to the original PyTorch Classifier.
    """
    embed_dim: int
    n_classes: int

    @nn.compact
    def __call__(self, x):
        # Get CLS token (first token in the sequence)
        # Input x shape: (B, S, E)
        cls_token_embedding = x[:, 0, :] # (B, E)

        # Linear classification layer
        logits = nn.Dense(features=self.n_classes, name='classifier_head')(cls_token_embedding) # (B, n_classes)
        return logits


class VisionTransformer(nn.Module):
    """
    Vision Transformer model.
    Equivalent to the original PyTorch VisionTransformer.
    """
    n_channels: int         # Number of input channels (e.g., 3 for RGB)
    embed_dim: int          # Embedding dimension
    n_layers: int           # Number of encoder layers
    n_attention_heads: int  # Number of attention heads
    forward_mul: int        # Multiplier for the feedforward hidden dimension
    image_size: int         # Input image size (assuming square images)
    patch_size: int         # Size of the image patches (assuming square patches)
    n_classes: int          # Number of output classes
    dropout_rate: float = 0.1 # Dropout rate
    pos_embed: str = 'learn' # Type of positional embedding
    max_relative_distance: int = 2 # Max relative distance for 'relative' pos embed
    string_type: str = 'cayley' # Type of STRING implementation

    def setup(self):
        # Calculate sequence length (number of patches + CLS token)
        n_patches_per_dim = self.image_size // self.patch_size
        n_patches = n_patches_per_dim * n_patches_per_dim
        self.seq_len = n_patches + 1

        # Embedding layer selection
        # Pass dropout_rate during the __call__ method, not initialization
        if self.pos_embed == 'learn':
            self.embedding = EmbedLayerWithLearn(
                n_channels=self.n_channels,
                embed_dim=self.embed_dim,
                image_size=self.image_size,
                patch_size=self.patch_size,
                # dropout_rate=self.dropout_rate, # Removed from init
                name='embedding_layer'
            )
        elif self.pos_embed == 'sinusoidal':
            self.embedding = EmbedLayerWithSinusoidal(
                n_channels=self.n_channels,
                embed_dim=self.embed_dim,
                image_size=self.image_size,
                patch_size=self.patch_size,
                # dropout_rate=self.dropout_rate, # Removed from init
                name='embedding_layer'
            )
        else: # 'none', 'relative', 'rope', 'string' use EmbedLayerWithNone for patch + CLS
             self.embedding = EmbedLayerWithNone(
                n_channels=self.n_channels,
                embed_dim=self.embed_dim,
                image_size=self.image_size,
                patch_size=self.patch_size,
                # dropout_rate=self.dropout_rate, # Removed from init
                name='embedding_layer'
            )

        # Encoder layers
        # Create a list of Encoder modules
        self.encoder_layers = [
            Encoder(
                embed_dim=self.embed_dim,
                n_attention_heads=self.n_attention_heads,
                forward_mul=self.forward_mul,
                seq_len=self.seq_len,
                dropout_rate=self.dropout_rate,
                pos_embed=self.pos_embed,
                max_relative_distance=self.max_relative_distance,
                string_type=self.string_type,
                name=f'encoder_layer_{i}'
            ) for i in range(self.n_layers)
        ]

        # Final normalization layer after the last block
        self.norm = nn.LayerNorm(name='final_norm')

        # Classifier head
        self.classifier = Classifier(
            embed_dim=self.embed_dim,
            n_classes=self.n_classes,
            name='classifier_head'
        )

    @nn.compact
    def __call__(self, x, train: bool):
        # x shape: (B, H, W, C) - JAX/Flax typically uses channel last for images

        # Patch and Positional Embedding
        # Pass dropout_rate here
        x = self.embedding(x, train=train, dropout_rate=self.dropout_rate)

        # Encoder blocks
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, train=train) # Dropout is handled inside Encoder

        # Final normalization
        x = self.norm(x)

        # Classification head
        # Input shape: (B, S, E), output shape: (B, n_classes)
        logits = self.classifier(x)

        return logits

# Note: JAX/Flax handle weight initialization within the Linen modules
# using initializers specified in the param definitions.
# The vit_init_weights function from PyTorch is not needed here.
