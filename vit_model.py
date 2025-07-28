# filename: vit_model.py

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# Import existing positional encoding modules
from positional_encodings.pos_embed_none import EmbedLayerWithNone
from positional_encodings.pos_embed_learn import EmbedLayerWithLearn
from positional_encodings.pos_embed_sinusoidal import EmbedLayerWithSinusoidal
from positional_encodings.pos_embed_relative import SelfAttentionWithRelative
from positional_encodings.pos_embed_rope import SelfAttentionWithRope
from positional_encodings.pos_embed_string import SelfAttentionWithString

# Import new 3D STRING implementation
from positional_encodings.pos_embed_string_3d import SelfAttentionWithString3D, EmbedLayerWithString3D

# Import new Uniform RoPE implementation
from positional_encodings.pos_embed_uniform_rope import SelfAttentionWithUniformRope


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
    """Original Self-attention module without positional encoding modifications."""
    embed_dim: int
    n_attention_heads: int

    def setup(self):
        self.head_embed_dim = self.embed_dim // self.n_attention_heads
        if self.head_embed_dim * self.n_attention_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by n_attention_heads")

        self.queries = nn.Dense(features=self.embed_dim, use_bias=False, name='queries')
        self.keys = nn.Dense(features=self.embed_dim, use_bias=False, name='keys')
        self.values = nn.Dense(features=self.embed_dim, use_bias=False, name='values')
        self.out_projection = nn.Dense(features=self.embed_dim, name='out_projection')

    @nn.compact
    def __call__(self, x, depth_map=None):
        b, s, e = x.shape

        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xq = jnp.transpose(xq, (0, 2, 1, 3))
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = jnp.transpose(xk, (0, 2, 1, 3))
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = jnp.transpose(xv, (0, 2, 1, 3))

        xk_t = jnp.transpose(xk, (0, 1, 3, 2))
        x_attention = jnp.matmul(xq, xk_t)
        x_attention = x_attention / (self.head_embed_dim ** 0.5)
        x_attention = jax.nn.softmax(x_attention, axis=-1)
        x = jnp.matmul(x_attention, xv)

        x = jnp.transpose(x, (0, 2, 1, 3))
        x = jnp.reshape(x, (b, s, e))
        x = self.out_projection(x)
        return x


class Encoder(nn.Module):
    """Transformer Encoder block with support for all positional encodings."""
    embed_dim: int
    n_attention_heads: int
    forward_mul: int
    seq_len: int
    image_size: int
    patch_size: int
    dropout_rate: float = 0.0
    pos_embed: str = 'learn'
    max_relative_distance: int = 2
    string_type: str = 'cayley'
    use_depth: bool = False

    @nn.compact
    def __call__(self, x, depth_map=None, train: bool = True):
        norm1_out = nn.LayerNorm(name='norm1')(x)

        if self.pos_embed == 'relative':
            attention_out = SelfAttentionWithRelative(
                embed_dim=self.embed_dim, n_attention_heads=self.n_attention_heads,
                seq_len=self.seq_len, max_relative_dist=self.max_relative_distance,
                name='attention')(norm1_out)
        elif self.pos_embed == 'rope':
            attention_out = SelfAttentionWithRope(
                embed_dim=self.embed_dim, n_attention_heads=self.n_attention_heads,
                seq_len=self.seq_len, name='attention')(norm1_out)
        elif self.pos_embed == 'string':
            attention_out = SelfAttentionWithString(
                embed_dim=self.embed_dim, n_attention_heads=self.n_attention_heads,
                seq_len=self.seq_len, string_type=self.string_type,
                name='attention')(norm1_out)
        elif self.pos_embed == 'string3d':
            attention_out = SelfAttentionWithString3D(
                embed_dim=self.embed_dim, n_attention_heads=self.n_attention_heads,
                seq_len=self.seq_len, image_size=self.image_size, patch_size=self.patch_size,
                string_type=self.string_type, use_depth=self.use_depth,
                name='attention')(norm1_out, depth_map)
        elif self.pos_embed == 'uniform_rope':
            attention_out = SelfAttentionWithUniformRope(
                embed_dim=self.embed_dim, n_attention_heads=self.n_attention_heads,
                seq_len=self.seq_len, name='attention')(norm1_out)
        else:  # 'none', 'learn', 'sinusoidal'
            attention_out = OriginalSelfAttention(
                embed_dim=self.embed_dim, n_attention_heads=self.n_attention_heads,
                name='attention')(norm1_out, depth_map)

        x = x + nn.Dropout(rate=self.dropout_rate, deterministic=not train)(attention_out)
        norm2_out = nn.LayerNorm(name='norm2')(x)
        fc1_out = nn.Dense(features=self.embed_dim * self.forward_mul, name='fc1')(norm2_out)
        activation_out = jax.nn.gelu(fc1_out)
        fc2_out = nn.Dense(features=self.embed_dim, name='fc2')(activation_out)
        x = x + nn.Dropout(rate=self.dropout_rate, deterministic=not train)(fc2_out)

        return x


class Classifier(nn.Module):
    """Classifier head for Vision Transformer."""
    embed_dim: int
    n_classes: int

    @nn.compact
    def __call__(self, x):
        cls_token_embedding = x[:, 0, :]
        logits = nn.Dense(features=self.n_classes, name='classifier_head')(cls_token_embedding)
        return logits


class VisionTransformer3D(nn.Module):
    """Vision Transformer model with 3D STRING and Uniform RoPE support."""
    n_channels: int = 3
    embed_dim: int = 128
    n_layers: int = 6
    n_attention_heads: int = 4
    forward_mul: int = 2
    image_size: int = 32
    patch_size: int = 4
    n_classes: int = 10
    dropout_rate: float = 0.1
    pos_embed: str = 'learn'
    max_relative_distance: int = 2
    string_type: str = 'cayley'
    use_depth: bool = False

    def setup(self):
        n_patches_per_dim = self.image_size // self.patch_size
        self.seq_len = (n_patches_per_dim ** 2) + 1

        if self.pos_embed == 'string3d':
            self.embedding = EmbedLayerWithString3D(
                n_channels=self.n_channels, embed_dim=self.embed_dim,
                image_size=self.image_size, patch_size=self.patch_size,
                string_type=self.string_type, use_depth=self.use_depth,
                name='embedding_layer')
        elif self.pos_embed == 'learn':
            self.embedding = EmbedLayerWithLearn(
                n_channels=self.n_channels, embed_dim=self.embed_dim,
                image_size=self.image_size, patch_size=self.patch_size,
                name='embedding_layer')
        elif self.pos_embed == 'sinusoidal':
            self.embedding = EmbedLayerWithSinusoidal(
                n_channels=self.n_channels, embed_dim=self.embed_dim,
                image_size=self.image_size, patch_size=self.patch_size,
                name='embedding_layer')
        else:  # 'none', 'relative', 'rope', 'string', 'uniform_rope'
            self.embedding = EmbedLayerWithNone(
                n_channels=self.n_channels, embed_dim=self.embed_dim,
                image_size=self.image_size, patch_size=self.patch_size,
                name='embedding_layer')

        self.encoder_layers = [
            Encoder(
                embed_dim=self.embed_dim, n_attention_heads=self.n_attention_heads,
                forward_mul=self.forward_mul, seq_len=self.seq_len,
                image_size=self.image_size, patch_size=self.patch_size,
                dropout_rate=self.dropout_rate, pos_embed=self.pos_embed,
                max_relative_distance=self.max_relative_distance,
                string_type=self.string_type, use_depth=self.use_depth,
                name=f'encoder_layer_{i}') for i in range(self.n_layers)
        ]

        self.norm = nn.LayerNorm(name='final_norm')
        self.classifier = Classifier(embed_dim=self.embed_dim, n_classes=self.n_classes, name='classifier_head')

    @nn.compact
    def __call__(self, x, depth_map=None, train: bool = True):
        if self.pos_embed == 'string3d':
            x = self.embedding(x, depth_map, train=train, dropout_rate=self.dropout_rate)
        else:
            x = self.embedding(x, train=train, dropout_rate=self.dropout_rate)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, depth_map, train=train)

        x = self.norm(x)
        logits = self.classifier(x)
        return logits


class VisionTransformer(VisionTransformer3D):
    """Original VisionTransformer class for backward compatibility."""
    def __init__(self, **kwargs):
        kwargs.pop('use_depth', None)
        super().__init__(**kwargs)
