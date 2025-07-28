# -*- coding: utf-8 -*-
"""
JAX-based Solver for Vision Transformer Training.
This version is corrected to handle both 2D and 3D (with depth) inputs
and follows JAX best practices by marking control-flow arguments as static.
"""
import os
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import unfreeze
from functools import partial  # Import partial for static arguments

# Import model and data loader
from vit_model import VisionTransformer3D
from data_loader import get_loader, get_batch_data

# Helper function to calculate accuracy
def calculate_accuracy(logits, labels):
    """Calculates prediction accuracy."""
    predicted_class = jnp.argmax(logits, axis=-1)
    true_class = jnp.argmax(labels, axis=-1)
    return jnp.mean(predicted_class == true_class)

class TrainState(train_state.TrainState):
    """Custom TrainState to handle batch stats if using BatchNorm."""
    batch_stats: dict = None

# ---------------------------------------------------------------------------------
# JIT-Compiled Pure Functions
# ---------------------------------------------------------------------------------

# *** CORRECTION ***: Mark 'use_depth' as a static argument
@partial(jax.jit, static_argnames=['use_depth'])
def _train_step(state, batch, dropout_rng, use_depth):
    """Performs a single training step as a pure function."""
    images, depth_maps, labels = get_batch_data(batch, use_depth=use_depth)

    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            images,
            depth_map=depth_maps,
            train=True,
            rngs={'dropout': dropout_rng},
            mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state['batch_stats'])

    acc = calculate_accuracy(logits, labels)
    return state, loss, acc

# *** CORRECTION ***: Mark 'use_depth' as a static argument
@partial(jax.jit, static_argnames=['use_depth'])
def _eval_step(state, batch, use_depth):
    """Performs a single evaluation step as a pure function."""
    images, depth_maps, labels = get_batch_data(batch, use_depth=use_depth)

    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        images,
        depth_map=depth_maps,
        train=False
    )
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    acc = calculate_accuracy(logits, labels)
    return loss, acc

# ---------------------------------------------------------------------------------
# Solver Class
# ---------------------------------------------------------------------------------

class Solver:
    def __init__(self, args):
        self.args = args
        self.use_depth = (args.pos_embed == 'string3d')

        self.key = jax.random.PRNGKey(args.seed)
        print("Loading data...")
        self.train_loader, self.test_loader = get_loader(args)

        print("Initializing model...")
        self.model = VisionTransformer3D(
            n_channels=3, embed_dim=128, n_layers=6, n_attention_heads=4,
            forward_mul=2, image_size=args.image_size, patch_size=args.patch_size,
            n_classes=args.n_classes, dropout_rate=args.dropout, pos_embed=args.pos_embed,
            max_relative_distance=args.max_relative_distance, string_type=args.string_type,
            use_depth=self.use_depth
        )
        self.state = self._create_train_state()
        self.history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    def _create_train_state(self):
        """Creates an initial TrainState."""
        self.key, init_key = jax.random.split(self.key)
        dummy_image = jnp.ones([1, self.args.image_size, self.args.image_size, 3])
        dummy_depth = jnp.ones([1, self.args.image_size, self.args.image_size, 1]) if self.use_depth else None

        variables = self.model.init({'params': init_key, 'dropout': init_key}, dummy_image, dummy_depth, train=False)
        params = variables['params']
        batch_stats = variables.get('batch_stats', {})

        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = len(self.train_loader) * self.args.warmup_epochs
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=self.args.lr,
            warmup_steps=warmup_steps, decay_steps=total_steps - warmup_steps, end_value=1e-6
        )
        tx = optax.adamw(learning_rate=schedule, weight_decay=1e-4)

        return TrainState.create(apply_fn=self.model.apply, params=params, tx=tx, batch_stats=batch_stats)

    def train(self):
        """Main training loop."""
        print("Starting training...")
        # Override epochs if specified for a single run
        epochs_to_run = self.args.epochs
        for epoch in range(epochs_to_run):
            train_loss, train_acc = 0.0, 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs_to_run} [Train]")
            for batch in pbar:
                self.key, dropout_key = jax.random.split(self.key)
                self.state, loss, acc = _train_step(self.state, batch, dropout_key, self.use_depth)
                train_loss += loss
                train_acc += acc
                pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")

            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_acc = train_acc / len(self.train_loader)

            test_loss, test_acc = 0.0, 0.0
            pbar_eval = tqdm(self.test_loader, desc=f"Epoch {epoch+1}/{epochs_to_run} [Eval]")
            for batch in pbar_eval:
                loss, acc = _eval_step(self.state, batch, self.use_depth)
                test_loss += loss
                test_acc += acc
                pbar_eval.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")

            avg_test_loss = test_loss / len(self.test_loader)
            avg_test_acc = test_acc / len(self.test_loader)

            print(
                f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
                f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}"
            )
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            self.history['test_loss'].append(avg_test_loss)
            self.history['test_acc'].append(avg_test_acc)

            checkpoints.save_checkpoint(
                ckpt_dir=self.args.model_path,
                target=unfreeze(self.state.params),
                step=epoch,
                prefix=f'{self.args.pos_embed}_',
                overwrite=True
            )

    def plot_graphs(self):
        """Plots and saves training history."""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['test_loss'], label='Test Loss')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['test_acc'], label='Test Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.args.output_path, 'training_graphs.png')
        plt.savefig(save_path)
        print(f"Training graphs saved to {save_path}")
        plt.show()
