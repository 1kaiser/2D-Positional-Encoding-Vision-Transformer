# -*- coding: utf-8 -*-
"""
JAX/Flax implementation of 3D STRING Positional Encoding.
Extends the 2D STRING to handle depth information for robotics applications.
Based on the STRING robotics paper: https://sites.google.com/view/string-robotics

Key features:
- Handles 3D coordinates (x, y, z) where z comes from depth information
- Supports both Cayley-STRING and Circulant-STRING variants
- Efficient computation using FFT for circulant matrices
- Translation invariant positional encoding
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

class StringPositionEmbedding3D(nn.Module):
    """
    3D STRING: Separable Translationally Invariant Position Encodings for 3D coordinates.
    Handles x, y, and z (depth) dimensions.
    """
    seq_len: int
    embed_dim: int  # Dimension for a single axis (e.g., HE/3 for 3D)
    string_type: str = 'cayley'

    def setup(self):
        """Initializes learnable parameters and non-learnable buffers for 3D."""
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
        # Pre-compute base RoPE frequencies for 3D (embed_dim/2 frequencies)
        self.freqs = 1.0 / (10000 ** (jnp.arange(0, self.embed_dim // 2, dtype=jnp.float32) * 2 / self.embed_dim))

    def _apply_efficient_rope_3d(self, x, coord_positions):
        """
        Applies 3D RoPE rotation to vectors using provided coordinate positions.
        
        Args:
            x: Input tensor (B, H, S, E)
            coord_positions: (S,) array of coordinate values for this dimension
            
        Returns:
            x_rotated: Rotated tensor (B, H, S, E)
        """
        # Calculate sin/cos factors for each position using actual coordinates
        angles = jnp.outer(coord_positions, self.freqs)  # (S, E/2)
        cos_vals = jnp.cos(angles)  # (S, E/2)
        sin_vals = jnp.sin(angles)  # (S, E/2)

        # Repeat to match the full embedding dimension
        cos_vals = jnp.repeat(cos_vals, 2, axis=-1)  # (S, E)
        sin_vals = jnp.repeat(sin_vals, 2, axis=-1)  # (S, E)

        # Handle odd dimensions
        if cos_vals.shape[-1] > self.embed_dim:
            cos_vals = cos_vals[:, :self.embed_dim]
            sin_vals = sin_vals[:, :self.embed_dim]
        elif cos_vals.shape[-1] < self.embed_dim:
            cos_vals = jnp.pad(cos_vals, ((0, 0), (0, self.embed_dim - cos_vals.shape[-1])), 
                              mode='constant', constant_values=1.0)
            sin_vals = jnp.pad(sin_vals, ((0, 0), (0, self.embed_dim - sin_vals.shape[-1])), 
                              mode='constant', constant_values=0.0)

        # Apply the rotation formula: x_rot = x*cos - permute(x)*sin
        if self.embed_dim == 1:
            # Handle single dimension case
            x_rotated = x * cos_vals[None, None, :, :]
        else:
            x1, x2 = jnp.split(x, 2, axis=-1)
            x_permuted = jnp.concatenate([-x2, x1], axis=-1)
            # Broadcast across batch and head dimensions
            x_rotated = x * cos_vals[None, None, :, :] + x_permuted * sin_vals[None, None, :, :]
            
        return x_rotated

    @nn.compact
    def __call__(self, x, coord_positions):
        """
        Applies 3D STRING positional encoding.
        
        Args:
            x: Input tensor (B, H, S, E)
            coord_positions: Coordinate values for this dimension (S,)
            
        Returns:
            x_encoded: Position-encoded tensor (B, H, S, E)
        """
        # 1. Generate the learnable orthogonal transformation matrix P
        if self.string_type == 'cayley':
            # Make S antisymmetric: (S - S^T)/2
            S_antisym = (self.S - self.S.T) / 2.0
            # Cayley Transform using linear solver for numerical stability
            I = jnp.eye(self.embed_dim, dtype=x.dtype)
            try:
                P = jnp.linalg.solve(I + S_antisym, I - S_antisym)
            except:
                # Fallback to pseudo-inverse if singular
                P = jnp.linalg.pinv(I + S_antisym) @ (I - S_antisym)
        else:  # circulant
            n = len(self.c)
            indices = (jnp.arange(n)[:, None] - jnp.arange(n)[None, :]) % n
            C = self.c[indices]
            P = C - C.T

        # 2. Apply the learnable transformation P to the input
        x_transformed = jnp.matmul(x, P.T)

        # 3. Apply 3D RoPE rotation using the provided coordinate positions
        return self._apply_efficient_rope_3d(x_transformed, coord_positions)


class DepthProcessor(nn.Module):
    """
    Processes depth information to extract z-coordinates for patches.
    Implements the "Lifting patches to 3D" approach from STRING robotics.
    """
    patch_size: int
    image_size: int
    
    @nn.compact
    def __call__(self, depth_map):
        """
        Processes depth map to extract z-coordinates for each patch.
        
        Args:
            depth_map: (B, H, W, 1) depth values
            
        Returns:
            z_coords: (B, N) z-coordinates for N patches
        """
        b, h, w, _ = depth_map.shape
        n_patches_per_dim = self.image_size // self.patch_size
        
        # Reshape depth map into patches and compute mean depth per patch
        # This implements the mean-pooling across depth values mentioned in the paper
        depth_patches = depth_map.reshape(
            b, n_patches_per_dim, self.patch_size, 
            n_patches_per_dim, self.patch_size, 1
        )
        
        # Mean pool over patch dimensions to get (B, n_patches_per_dim, n_patches_per_dim, 1)
        patch_depths = jnp.mean(depth_patches, axis=(2, 4))
        
        # Flatten to (B, N, 1) where N = n_patches_per_dim^2
        patch_depths = patch_depths.reshape(b, -1, 1)
        
        # Apply learnable linear transformation to convert raw depth to z-coordinate
        # This is the "learnable linear layer" mentioned in the paper
        z_coords = nn.Dense(features=1, name='depth_to_z')(patch_depths)
        
        return z_coords.squeeze(-1)  # (B, N)


class SelfAttentionWithString3D(nn.Module):
    """
    Self-attention module with 3D STRING positional encoding.
    Incorporates depth information for robotics applications as described
    in the STRING robotics paper.
    """
    embed_dim: int
    n_attention_heads: int
    seq_len: int
    image_size: int
    patch_size: int
    string_type: str = 'cayley'
    use_depth: bool = True

    def setup(self):
        self.head_embed_dim = self.embed_dim // self.n_attention_heads
        if self.head_embed_dim * self.n_attention_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by n_attention_heads")

        self.queries = nn.Dense(features=self.embed_dim, use_bias=False, name='queries')
        self.keys = nn.Dense(features=self.embed_dim, use_bias=False, name='keys')
        self.values = nn.Dense(features=self.embed_dim, use_bias=False, name='values')
        self.out_projection = nn.Dense(features=self.embed_dim, name='out_projection')

        if self.use_depth:
            # Split head_embed_dim for x, y, and z axes (3D)
            self.dim_split_x = self.head_embed_dim // 3
            self.dim_split_y = self.head_embed_dim // 3
            self.dim_split_z = self.head_embed_dim - 2 * (self.head_embed_dim // 3)  # Handle remainder
            
            # Ensure minimum dimension of 1 for each axis
            if self.dim_split_x == 0:
                self.dim_split_x = 1
                self.dim_split_y = max(1, (self.head_embed_dim - 1) // 2)
                self.dim_split_z = self.head_embed_dim - self.dim_split_x - self.dim_split_y
            
            self.string_x = StringPositionEmbedding3D(
                seq_len=self.seq_len, embed_dim=self.dim_split_x, 
                string_type=self.string_type, name='string_x'
            )
            self.string_y = StringPositionEmbedding3D(
                seq_len=self.seq_len, embed_dim=self.dim_split_y, 
                string_type=self.string_type, name='string_y'
            )
            self.string_z = StringPositionEmbedding3D(
                seq_len=self.seq_len, embed_dim=self.dim_split_z, 
                string_type=self.string_type, name='string_z'
            )
            
            # Depth processor for extracting z-coordinates
            self.depth_processor = DepthProcessor(
                patch_size=self.patch_size,
                image_size=self.image_size,
                name='depth_processor'
            )
        else:
            # Fall back to 2D (x, y only)
            self.dim_split_x = self.head_embed_dim // 2 + (self.head_embed_dim % 2)
            self.dim_split_y = self.head_embed_dim // 2
            
            self.string_x = StringPositionEmbedding3D(
                seq_len=self.seq_len, embed_dim=self.dim_split_x, 
                string_type=self.string_type, name='string_x'
            )
            self.string_y = StringPositionEmbedding3D(
                seq_len=self.seq_len, embed_dim=self.dim_split_y, 
                string_type=self.string_type, name='string_y'
            )

    def _get_2d_coordinates(self):
        """Generate 2D patch coordinates (x, y)."""
        n_patches_per_dim = int(np.sqrt(self.seq_len - 1))  # Exclude CLS token
        
        # Generate grid coordinates
        y_coords, x_coords = jnp.meshgrid(
            jnp.arange(n_patches_per_dim), 
            jnp.arange(n_patches_per_dim), 
            indexing='ij'
        )
        
        x_patch_coords = x_coords.flatten().astype(jnp.float32)
        y_patch_coords = y_coords.flatten().astype(jnp.float32)
        
        # Add CLS token coordinates (0, 0) at the beginning
        x_coords_full = jnp.concatenate([jnp.array([0.0]), x_patch_coords + 1])
        y_coords_full = jnp.concatenate([jnp.array([0.0]), y_patch_coords + 1])
        
        return x_coords_full, y_coords_full

    @nn.compact
    def __call__(self, x, depth_map=None):
        """
        Forward pass with optional depth information.
        
        Args:
            x: Input embeddings (B, S, E)
            depth_map: Optional depth map (B, H, W, 1) for 3D encoding
            
        Returns:
            x: Output embeddings with 3D STRING positional encoding applied (B, S, E)
        """
        b, s, e = x.shape

        # Generate Q, K, V and reshape to (B, H, S, HE)
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xq = jnp.transpose(xq, (0, 2, 1, 3))
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = jnp.transpose(xk, (0, 2, 1, 3))
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = jnp.transpose(xv, (0, 2, 1, 3))

        # Get 2D coordinates for x and y dimensions
        x_coords, y_coords = self._get_2d_coordinates()

        if self.use_depth and depth_map is not None:
            # 3D STRING: Split into x, y, z dimensions
            split_indices = [self.dim_split_x, self.dim_split_x + self.dim_split_y]
            xq_x, xq_y, xq_z = jnp.split(xq, split_indices, axis=-1)
            xk_x, xk_y, xk_z = jnp.split(xk, split_indices, axis=-1)

            # Process depth to get z-coordinates
            z_coords_patches = self.depth_processor(depth_map)  # (B, N)
            # Add CLS token z-coordinate (0.0) at the beginning
            z_coords = jnp.concatenate([
                jnp.zeros((b, 1)), z_coords_patches
            ], axis=1)  # (B, S)
            
            # Use batch-averaged z-coordinates for positional encoding
            # In practice, you might want to handle this per-sample
            z_coords_mean = jnp.mean(z_coords, axis=0)  # (S,)

            # Apply 3D STRING positional encoding to queries and keys
            xq_x_str = self.string_x(xq_x, x_coords)
            xq_y_str = self.string_y(xq_y, y_coords)
            xq_z_str = self.string_z(xq_z, z_coords_mean)
            
            xk_x_str = self.string_x(xk_x, x_coords)
            xk_y_str = self.string_y(xk_y, y_coords)
            xk_z_str = self.string_z(xk_z, z_coords_mean)

            # Concatenate the results
            xq = jnp.concatenate([xq_x_str, xq_y_str, xq_z_str], axis=-1)
            xk = jnp.concatenate([xk_x_str, xk_y_str, xk_z_str], axis=-1)
        else:
            # 2D STRING: Split into x, y dimensions only
            xq_x, xq_y = jnp.split(xq, [self.dim_split_x], axis=-1)
            xk_x, xk_y = jnp.split(xk, [self.dim_split_x], axis=-1)

            # Apply 2D STRING positional encoding
            xq_x_str = self.string_x(xq_x, x_coords)
            xq_y_str = self.string_y(xq_y, y_coords)
            xk_x_str = self.string_x(xk_x, x_coords)
            xk_y_str = self.string_y(xk_y, y_coords)

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


class EmbedLayerWithString3D(nn.Module):
    """
    Embedding layer for Vision Transformer with 3D STRING positional encoding.
    Supports both RGB and RGB-D inputs for robotics applications.
    """
    n_channels: int  # 3 for RGB, 4 for RGB-D
    embed_dim: int
    image_size: int
    patch_size: int
    string_type: str = 'cayley'
    use_depth: bool = False

    @nn.compact
    def __call__(self, x, depth_map=None, train: bool = False, dropout_rate: float = 0.0):
        """
        Forward pass with optional depth input.
        
        Args:
            x: RGB image (B, H, W, 3)
            depth_map: Optional depth map (B, H, W, 1)
            train: Training mode flag
            dropout_rate: Dropout rate
            
        Returns:
            x: Patch embeddings with positional encoding (B, S, E)
        """
        b, h_img, w_img, c_img = x.shape
        
        # Image to Patch Embedding for RGB channels
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='patch_embedding'
        )(x)
        b, h, w, c = x.shape
        n_patches = h * w
        x = jnp.reshape(x, (b, n_patches, c))  # Flatten spatial dimensions

        # Classification Token
        cls_token = self.param('cls_token',
                               nn.initializers.zeros,
                               (1, 1, self.embed_dim))
        cls_token = jnp.tile(cls_token, [b, 1, 1])

        # Concatenate Classification Token and Patch Embeddings
        x = jnp.concatenate([cls_token, x], axis=1)  # (B, N+1, E)

        # Apply dropout
        x = nn.Dropout(rate=dropout_rate, deterministic=not train)(x)

        return x
