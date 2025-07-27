import jax
import jax.numpy as jnp
import numpy as np

def print_args(args):
    """Prints the arguments."""
    print("\n--- Arguments ---")
    for arg in vars(args):
        print(f"('{arg}', {getattr(args, arg)})")
    print("-----------------\n")

# Add the positional helper functions back, using JAX/NumPy
def get_x_positions(n_patches_per_dim):
    """Generates x positions for a square grid using JAX/NumPy."""
    x_pos = jnp.arange(n_patches_per_dim)
    x_pos = jnp.repeat(x_pos, n_patches_per_dim)
    # Add position for CLS token (position 0)
    return jnp.concatenate([jnp.array([0]), x_pos + 1]) # Start patch positions from 1

def get_y_positions(n_patches_per_dim):
    """Generates y positions for a square grid using JAX/NumPy."""
    y_pos = jnp.arange(n_patches_per_dim)
    y_pos = jnp.tile(y_pos, n_patches_per_dim)
    # Add position for CLS token (position 0)
    return jnp.concatenate([jnp.array([0]), y_pos + 1]) # Start patch positions from 1
