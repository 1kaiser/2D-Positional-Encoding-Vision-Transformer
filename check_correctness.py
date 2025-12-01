
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import sys
import os

# Import modules to be tested
from vit_model import VisionTransformer

def run_training_step(pos_embed_type, string_type='cayley', use_depth=False):
    print(f"  Testing {pos_embed_type}...", end=' ')

    # Model config
    model = VisionTransformer(
        n_channels=3,
        embed_dim=64,
        n_layers=2,
        n_attention_heads=4,
        image_size=32,
        patch_size=8,
        n_classes=10,
        pos_embed=pos_embed_type,
        string_type=string_type,
        use_depth=use_depth,
        dropout_rate=0.1
    )

    # Random data
    key = jax.random.PRNGKey(0)
    key, init_key, train_key = jax.random.split(key, 3)

    x = jax.random.normal(key, (2, 32, 32, 3))
    y = jax.random.randint(key, (2,), 0, 10)
    depth_map = None
    if pos_embed_type == 'string3d' or use_depth:
         depth_map = jax.random.normal(key, (2, 32, 32, 1))

    # Init
    variables = model.init({'params': init_key, 'dropout': init_key}, x, depth_map=depth_map, train=True)
    params = variables['params']

    # Optimizer
    tx = optax.adam(learning_rate=0.001)
    opt_state = tx.init(params)

    # Train step function
    @jax.jit
    def train_step(params, opt_state, x, y, depth_map, key):
        # Fix: opt_state.count is a method/property mismatch or just cast issue.
        # Actually opt_state is a named tuple, .count might be specific to some optimizers or None.
        # optax.adam state is typically (count, mu, nu).
        # We can just use key splitting inside.

        # But wait, optax.adam state has 'count'.
        # However, `opt_state.count` might be a device array.
        # jax.random.fold_in expects a number or array.
        # The error said: TypeError: int() argument must be a string... not 'builtin_function_or_method'
        # This implies opt_state.count is a method?
        # Let's inspect opt_state structure or just ignore count and split key.

        dropout_key = jax.random.fold_in(key, 0) # Just fold in a constant or step

        def loss_fn(params):
            logits = model.apply(
                {'params': params},
                x, depth_map=depth_map, train=True,
                rngs={'dropout': dropout_key}
            )
            one_hot = jax.nn.one_hot(y, 10)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Run a few steps
    try:
        current_key = train_key
        for i in range(3):
            current_key, step_key = jax.random.split(current_key)
            params, opt_state, loss = train_step(params, opt_state, x, y, depth_map, step_key)
        print(f"PASS (Loss: {loss:.4f})")
        return True, loss
    except Exception as e:
        print(f"FAIL ({e})")
        import traceback
        traceback.print_exc()
        return False, None

def generate_visualization_report():
    print("\nGenerating visualization report...")
    report_lines = []
    report_lines.append("Positional Embedding Visualization Report")
    report_lines.append("======================================")

    # Helper to visualize embedding
    def viz_embed(pos_embed_type):
        model = VisionTransformer(
            n_channels=3,
            embed_dim=16,
            n_layers=1,
            n_attention_heads=4,
            image_size=16, # 4x4 patches
            patch_size=4,
            n_classes=10,
            pos_embed=pos_embed_type
        )
        key = jax.random.PRNGKey(42)
        x = jnp.zeros((1, 16, 16, 3))
        try:
            variables = model.init({'params': key, 'dropout': key}, x, train=False)
            report_lines.append(f"\n=== {pos_embed_type.upper()} ===")

            if pos_embed_type == 'learn':
                 pe = variables['params']['embedding_layer']['pos_embed']
                 report_lines.append(f"Shape: {pe.shape}")
                 sim = jnp.matmul(pe[0], pe[0].T)
                 report_lines.append("Similarity matrix (top-left 5x5):")
                 report_lines.append(str(np.array(sim[:5, :5])))
            elif pos_embed_type == 'relative':
                 table = variables['params']['encoder_layer_0']['attention']['relative_bias_table']
                 report_lines.append(f"Bias Table Shape: {table.shape}")
                 report_lines.append(f"First 5 values: {table[:5]}")
            else:
                 report_lines.append("Encoding is functional/dynamic (applied during attention).")
                 report_lines.append("Verified via runtime tests.")

        except Exception as e:
            report_lines.append(f"Error visualizing {pos_embed_type}: {e}")

    for enc in ['learn', 'sinusoidal', 'relative', 'rope', 'string']:
        viz_embed(enc)

    report_content = '\n'.join(report_lines)
    print(report_content)

    with open('check_correctness_report.txt', 'w') as f:
        f.write(report_content)
    print("\nReport saved to 'check_correctness_report.txt'")

def main():
    print("Running Correctness Checks & Small Training Tests")
    print("===============================================")

    encodings = [
        ('none', 'cayley', False),
        ('learn', 'cayley', False),
        ('sinusoidal', 'cayley', False),
        ('relative', 'cayley', False),
        ('rope', 'cayley', False),
        ('string', 'cayley', False),
        ('string', 'circulant', False),
        ('string3d', 'cayley', True),
        ('uniform_rope', 'cayley', False)
    ]

    all_passed = True
    for enc, string_type, use_depth in encodings:
        passed, _ = run_training_step(enc, string_type, use_depth)
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll training tests PASSED.")
    else:
        print("\nSome training tests FAILED.")

    generate_visualization_report()

if __name__ == "__main__":
    main()
