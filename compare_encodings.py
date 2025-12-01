
import os
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import numpy as np
import datetime
import argparse
from tqdm import tqdm

# Import local modules
from vit_model import VisionTransformer
from data_loader import get_loader, get_batch_data

class TrainState(train_state.TrainState):
    rng: jax.random.PRNGKey

def create_train_state(rng, config):
    model = VisionTransformer(
        n_channels=3,
        embed_dim=config.embed_dim,
        n_layers=config.n_layers,
        n_attention_heads=config.n_attention_heads,
        image_size=config.image_size,
        patch_size=config.patch_size,
        n_classes=config.n_classes,
        pos_embed=config.pos_embed,
        string_type=config.string_type,
        use_depth=config.use_depth,
        dropout_rate=config.dropout
    )

    # Initialize parameters
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_input = jnp.ones((1, config.image_size, config.image_size, 3))
    dummy_depth = None
    if config.use_depth:
        dummy_depth = jnp.ones((1, config.image_size, config.image_size, 1))

    variables = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_input, depth_map=dummy_depth, train=False)
    params = variables['params']

    tx = optax.adamw(learning_rate=config.lr, weight_decay=1e-4)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        rng=dropout_rng # Store RNG for dropout
    )

@jax.jit
def train_step(state, batch_images, batch_labels, batch_depth=None):
    dropout_key = jax.random.fold_in(state.rng, state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch_images,
            depth_map=batch_depth,
            train=True,
            rngs={'dropout': dropout_key}
        )
        # batch_labels is already one-hot from data_loader
        loss = optax.softmax_cross_entropy(logits, batch_labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    # Convert one-hot labels back to indices for accuracy
    gt_labels = jnp.argmax(batch_labels, -1)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)
    return state, loss, accuracy

@jax.jit
def eval_step(state, batch_images, batch_labels, batch_depth=None):
    logits = state.apply_fn(
        {'params': state.params},
        batch_images,
        depth_map=batch_depth,
        train=False
    )
    # batch_labels is already one-hot
    loss = optax.softmax_cross_entropy(logits, batch_labels).mean()

    gt_labels = jnp.argmax(batch_labels, -1)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)
    return loss, accuracy

def run_experiment(pos_embed, string_type='cayley', use_depth=False, epochs=3):
    print(f"\nTraining with {pos_embed} (string_type={string_type}, use_depth={use_depth})")
    print("-" * 60)

    class Config:
        dataset = 'cifar10'
        image_size = 32
        patch_size = 4
        n_classes = 10
        batch_size = 64 # Small batch size for speed/memory
        embed_dim = 64  # Small model
        n_layers = 2
        n_attention_heads = 4
        dropout = 0.0 # Disable dropout for short deterministic check stability
        lr = 1e-3
        pos_embed = ''
        string_type = ''
        use_depth = False
        depth_simulation = False # Will be set below
        depth_noise_std = 0.1

    config = Config()
    config.pos_embed = pos_embed
    config.string_type = string_type
    config.use_depth = use_depth
    if pos_embed == 'string3d':
        config.use_depth = True
    config.depth_simulation = config.use_depth

    # Load data
    train_loader, test_loader = get_loader(config)

    # Initialize state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config)

    # Train Loop
    train_metrics = []

    # We only run a few batches per epoch to speed up this "small test"
    steps_per_epoch = 20 # Limit steps

    for epoch in range(epochs):
        batch_losses = []
        batch_accs = []

        # Train
        step_count = 0
        for batch in train_loader:
            if step_count >= steps_per_epoch: break

            imgs, depth, lbls = get_batch_data(batch, config.use_depth)

            # Simple normalization check (dataset loader likely does it)
            # data_loader.py divides by 255.

            state, loss, acc = train_step(state, imgs, lbls, depth)
            batch_losses.append(loss)
            batch_accs.append(acc)
            step_count += 1

        train_loss = np.mean(batch_losses)
        train_acc = np.mean(batch_accs)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

    # Final Eval
    test_losses = []
    test_accs = []
    step_count = 0
    for batch in test_loader:
        if step_count >= steps_per_epoch: break
        imgs, depth, lbls = get_batch_data(batch, config.use_depth)
        loss, acc = eval_step(state, imgs, lbls, depth)
        test_losses.append(loss)
        test_accs.append(acc)
        step_count += 1

    final_acc = np.mean(test_accs)
    print(f"Final Test Acc: {final_acc:.4f}")
    return final_acc

def main():
    print("Running Comparative Training Test on CIFAR-10 (Subset)")
    print("===================================================")

    # List of configurations to test
    configs = [
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

    results = {}

    for pe, st, ud in configs:
        try:
            acc = run_experiment(pe, st, ud, epochs=3)
            key = f"{pe}"
            if pe == 'string': key += f"_{st}"
            if pe == 'string3d': key += f"_{st}"
            results[key] = acc
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[f"{pe}_FAILED"] = 0.0

    print("\n\nPerformance Comparison (3 Epochs, Small Model, 20 steps/epoch):")
    print("---------------------------------------------------------------")
    for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:20s}: {v*100:.2f}%")

    with open('performance_comparison.txt', 'w') as f:
        f.write("Performance Comparison (3 Epochs, Small Model, 20 steps/epoch):\n")
        f.write("---------------------------------------------------------------\n")
        for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{k:20s}: {v*100:.2f}%\n")

if __name__ == '__main__':
    main()
