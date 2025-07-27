import os
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_loader
from vit_model import VisionTransformer # Import the Flax ViT model
from sklearn.metrics import accuracy_score
import time # Import time for tracking duration
from flax.serialization import to_bytes # Import for saving

# Define a simple Flax TrainState
# This holds the model parameters, optimizer state, and step count.
class TrainState(train_state.TrainState):
    # Nothing to add beyond the base class for now
    pass

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.key = jax.random.PRNGKey(args.seed if hasattr(args, 'seed') else 0) # Use a seed if provided, otherwise default

        # Get data loaders (now yield NumPy arrays)
        self.train_loader, self.test_loader = get_loader(args)

        # Initialize the Flax model
        # Flax models are functional, so we need to initialize parameters.
        # We need a dummy input with the correct shape (B, H, W, C)
        # Batch size can be 1 for initialization.
        dummy_input = jnp.ones((1, args.image_size, args.image_size, 3), dtype=jnp.float32)
        # Need a separate key for model parameters initialization
        self.key, model_key = jax.random.split(self.key)
        # Need a separate dropout key for initialization
        self.key, init_dropout_key = jax.random.split(self.key)


        self.model = VisionTransformer(
            n_channels=3,
            embed_dim=128, # These hyperparameters match the PyTorch version
            n_layers=6,
            n_attention_heads=4,
            forward_mul=2,
            image_size=args.image_size,
            patch_size=args.patch_size,
            dropout_rate=args.dropout, # Assuming args has dropout, default 0.1
            n_classes=args.n_classes,
            pos_embed=args.pos_embed,
            max_relative_distance=args.max_relative_distance,
            string_type=getattr(args, 'string_type', 'cayley')
        )

        # Initialize model parameters. Pass dummy input and train=False for initialization
        # Pass dropout_key via rngs argument
        # Corrected: Pass rngs as a separate keyword argument
        # AFTER
        initial_variables = self.model.init({'params': model_key, 'dropout': init_dropout_key}, dummy_input, train=False)
        self.params = initial_variables['params'] # Extract learnable parameters



        # Define optimizer (using Optax)
        # AdamW is a common choice for Vision Transformers
        optimizer = optax.adamw(
            learning_rate=args.lr,
            weight_decay=1e-3 # Assuming a default weight decay
        )

        # Create TrainState
        self.state = TrainState.create(
            apply_fn=self.model.apply, # Function to apply parameters (the model's __call__)
            params=self.params,        # Initial parameters
            tx=optimizer               # Optimizer
        )

        # Flax does not have explicit `load_state_dict` in the same way as PyTorch
        # Loading would involve loading saved parameters into self.params or self.state.params
        # We will skip loading for now as per the subtask scope, but keep the check.
        if self.args.load_model:
             print("Loading pretrained models is not yet implemented for JAX conversion.")
             # In a full implementation, you would load parameters here:
             # loaded_params = ... load from file ...
             # self.state = self.state.replace(params=loaded_params)


        # Training loss function (Cross-Entropy)
        # In JAX, loss functions are typically plain functions
        self.loss_fn = optax.softmax_cross_entropy # For one-hot labels

        # Arrays to record training progression
        self.train_losses     = []
        self.test_losses      = []
        self.train_accuracies = []
        self.test_accuracies  = []

        # JIT-compile the step functions *once* during initialization
        self._test_step = jax.jit(self._test_step_impl, static_argnums=[2]) # Mark loss_fn as static (3rd arg, index 2)
        # Mark loss_fn as static for train_step
        self._train_step = jax.jit(self._train_step_impl, static_argnums=[3])


    # Helper implementation for the JIT-compiled test step
    def _test_step_impl(self, state, batch, loss_fn):
        """Single test step implementation."""
        images, labels = batch
        # Apply the model with parameters, dropout key (dummy for test), and train=False
        # Pass a dummy dropout key for evaluation via rngs argument
        logits = state.apply_fn({'params': state.params}, images, train=False, rngs={'dropout': jax.random.PRNGKey(0)})
        loss = jnp.mean(loss_fn(logits=logits, labels=labels)) # Loss for the batch
        predicted_labels = jnp.argmax(logits, axis=-1)
        true_labels = jnp.argmax(jnp.asarray(labels), axis=-1) # Convert one-hot back to integer for comparison
        accuracy = jnp.mean(predicted_labels == true_labels)
        return loss, accuracy

    def test_dataset(self, loader):
        """Evaluates the model on a dataset."""
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0

        # Iterate through the loader (yields NumPy arrays)
        for images, labels in loader:
            # Convert NumPy arrays to JAX arrays before passing to the JIT-compiled function
            images = jnp.asarray(images)
            labels = jnp.asarray(labels)

            # Call the JIT-compiled version
            batch_loss, batch_accuracy = self._test_step(self.state, (images, labels), self.loss_fn)

            total_loss += batch_loss * images.shape[0] # Accumulate weighted by batch size
            total_accuracy += batch_accuracy * images.shape[0] # Accumulate weighted by batch size
            num_samples += images.shape[0] # Accumulate total number of samples

        if num_samples == 0:
             return 0.0, 0.0 # Avoid division by zero

        avg_loss = total_loss / num_samples
        avg_accuracy = total_accuracy / num_samples

        return avg_accuracy, avg_loss

    # Helper implementation for the JIT-compiled train step
    def _train_step_impl(self, state, batch, dropout_key, loss_fn):
        """Single training step implementation."""
        images, labels = batch

        # Define the loss function for gradient computation
        # Pass loss_fn explicitly to compute_loss
        def compute_loss(params, loss_fn):
            # Apply the model with parameters, dropout key, and train=True
            # Pass dropout_key via rngs argument
            logits = state.apply_fn({'params': params}, images, train=True, rngs={'dropout': dropout_key})
            loss = jnp.mean(loss_fn(logits=logits, labels=labels)) # Mean loss over the batch
            return loss, logits # Return loss and logits for metrics

        # Compute gradients using JAX's value_and_grad
        # Now compute_loss takes loss_fn as an argument
        grad_fn = jax.value_and_grad(lambda params: compute_loss(params, loss_fn), has_aux=True) # has_aux=True because compute_loss returns (loss, logits)

        (loss, logits), grads = grad_fn(state.params)

        # Apply gradients and update the optimizer state
        new_state = state.apply_gradients(grads=grads)

        # Calculate metrics (accuracy)
        predicted_labels = jnp.argmax(logits, axis=-1)
        true_labels = jnp.argmax(jnp.asarray(labels), axis=-1) # Convert one-hot back to integer
        accuracy = jnp.mean(predicted_labels == true_labels)

        return new_state, loss, accuracy

    def train(self):
        iters_per_epoch = 0
        # Calculate iters_per_epoch by iterating through the loader once (or estimate)
        # This is needed for printing progress. Data loaders are iterators.
        # A simple way is to estimate based on dataset size and batch size.
        # CIFAR10 train size = 50000, test size = 10000
        if self.args.dataset == 'cifar10':
             dataset_size = 50000
        elif self.args.dataset == 'cifar100':
             dataset_size = 50000 # CIFAR100 train size is also 50000
        else:
             # Fallback estimate, or could iterate one epoch to count
             dataset_size = self.args.batch_size * 400 # A rough estimate

        iters_per_epoch = dataset_size // self.args.batch_size
        if dataset_size % self.args.batch_size != 0:
             iters_per_epoch += 1

        # Optax learning rate schedule
        # Use warmup_cosine_decay_schedule for combining warmup and cosine decay
        total_steps = self.args.epochs * iters_per_epoch
        warmup_steps = self.args.warmup_epochs * iters_per_epoch

        # Ensure decay_steps is positive and accounts for warmup steps
        # The cosine decay starts after warmup.
        cosine_decay_steps = total_steps - warmup_steps
        if cosine_decay_steps <= 0:
             # If warmup is as long as or longer than total epochs, just use warmup schedule
             print("Warning: Warmup epochs >= total epochs. Using only linear warmup.")
             schedule = optax.linear_schedule(
                 init_value=0.0,         # Start from 0 learning rate
                 end_value=self.args.lr, # Peak learning rate after warmup
                 transition_steps=total_steps # Warmup for the full duration
             )
        else:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,         # Start from 0 learning rate
                peak_value=self.args.lr, # Peak learning rate after warmup
                warmup_steps=warmup_steps,
                decay_steps=cosine_decay_steps,
                end_value=1e-5         # Minimum learning rate after decay
            )


        # Redefine the optimizer with the scheduled learning rate
        optimizer = optax.adamw(
            learning_rate=schedule, # Use the scheduled learning rate
            weight_decay=1e-3
        )
        # Recreate the state with the new optimizer
        self.state = TrainState.create(
            apply_fn=self.state.apply_fn, # Keep the current apply_fn
            params=self.state.params, # Keep the current parameters
            tx=optimizer
        )

        # Variable to capture best test accuracy
        best_acc = 0

        # Training loop
        print("Starting training...")
        start_time = time.time()

        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            # Create a new dropout key for each epoch
            self.key, dropout_key = jax.random.split(self.key)

            # Arrays to record epoch loss and accuracy
            train_epoch_loss_list = []
            train_epoch_accuracy_list = []

            # Loop on loader (yields NumPy arrays)
            # Re-initialize the loader iterator for each epoch
            train_loader_iter = get_loader(self.args)[0] # Get a new iterator

            for i, (x, y) in enumerate(train_loader_iter):
                # # <<<< ADD THIS BLOCK TO LIMIT ITERATIONS >>>>
                # if i >= 10:
                #     break
                # # <<<< END OF CHANGE >>>>
                # Convert NumPy arrays to JAX arrays
                x = jnp.asarray(x)
                y = jnp.asarray(y)

                # Perform training step using the JIT-compiled function
                # Call the JIT-compiled version
                self.state, loss, accuracy = self._train_step(self.state, (x, y), dropout_key, self.loss_fn)

                # Record batch metrics (convert JAX arrays to Python scalars)
                train_epoch_loss_list.append(float(loss))
                train_epoch_accuracy_list.append(float(accuracy))

                # Log training progress
                if i % 50 == 0 or i == (iters_per_epoch - 1):
                    # Optax schedule is applied internally by the optimizer based on state.step
                    # We can't directly access the 'current_lr' from the schedule object outside the optimizer apply.
                    # The learning rate is part of the optimizer state.
                    # To get the current LR, we can calculate it using the schedule function and the current step.
                    current_step = self.state.step
                    current_lr = schedule(current_step) # Calculate LR using the schedule function
                    print(f'Ep: {epoch+1}/{self.args.epochs}\tIt: {i+1}/{iters_per_epoch}\tbatch_loss: {loss:.4f}\tbatch_accuracy: {accuracy:.2%}\tlr: {current_lr:.6f}')

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")

            # Test the test set after every epoch
            # Test training set every 25 epochs as in original
            # Re-initialize the test loader iterator for each test run
            test_loader_iter = get_loader(self.args)[1] # Get a new iterator
            test_acc, test_loss = self.test_dataset(test_loader_iter)
            # Test train set conditionally
            if (epoch + 1) % 25 == 0:
                 train_loader_test_iter = get_loader(self.args)[0] # Get a new iterator
                 train_acc_test, train_loss_test = self.test_dataset(train_loader_test_iter)
                 print(f"Train acc: {train_acc_test:.2%}\tTrain loss: {train_loss_test:.4f}")

            print(f"Test acc: {test_acc:.2%}\tTest loss: {test_loss:.4f}")


            # Capture best test accuracy
            best_acc = max(test_acc, best_acc)
            print(f"Best test acc: {best_acc:.2%}\n")

            # Save model parameters (Flax saves parameters as a dictionary)
            # Create model path if it doesn't exist
            os.makedirs(self.args.model_path, exist_ok=True)
            model_filename = f"ViT_model_{self.args.pos_embed}"
            if self.args.pos_embed == 'string':
                model_filename += f"_{self.args.string_type}"
            model_filename += ".params" # Use .params extension for Flax parameters
            param_path = os.path.join(self.args.model_path, model_filename)

            # Save parameters using Flax's serialization
            # Convert parameters to a byte string
            param_bytes = to_bytes(self.state.params)

            # Save the byte string to a file
            with open(param_path, 'wb') as f:
                f.write(param_bytes)

            print(f"Model parameters saved to {param_path}")


            # Update training progression metric arrays
            self.train_losses.append(np.mean(train_epoch_loss_list))
            self.test_losses.append(test_loss)
            self.train_accuracies.append(np.mean(train_epoch_accuracy_list))
            self.test_accuracies.append(test_acc)

        end_time = time.time()
        total_duration = end_time - start_time
        print(f"\nTraining finished. Total duration: {total_duration:.2f} seconds.")


    def plot_graphs(self):
        """Plots training and test loss and accuracy."""
        # Create descriptive filename for plots
        plot_suffix = f"{self.args.pos_embed}"
        if self.args.pos_embed == 'string':
            plot_suffix += f"_{self.args.string_type}"

        # Ensure output directory exists
        os.makedirs(self.args.output_path, exist_ok=True)

        # Plot graph of loss values
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, color='b', label='Train')
        plt.plot(self.test_losses, color='r', label='Test')

        plt.ylabel('Loss', fontsize = 18)
        plt.yticks(fontsize=16)
        plt.xlabel('Epoch', fontsize = 18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)
        plt.title(f'Training Loss - {plot_suffix.upper()}', fontsize=16)

        # plt.show()  # Uncomment to display graph
        plt.savefig(os.path.join(self.args.output_path, f'graph_loss_{plot_suffix}.png'), bbox_inches='tight')
        plt.close('all')


        # Plot graph of accuracies
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracies, color='b', label='Train')
        plt.plot(self.test_accuracies, color='r', label='Test')

        plt.ylabel('Accuracy', fontsize = 18)
        plt.yticks(fontsize=16)
        plt.xlabel('Epoch', fontsize = 18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)
        plt.title(f'Training Accuracy - {plot_suffix.upper()}', fontsize=16)

        # plt.show()  # Uncomment to display graph
        plt.savefig(os.path.join(self.args.output_path, f'graph_accuracy_{plot_suffix}.png'), bbox_inches='tight')
        plt.close('all')

        # Print final results
        final_test_acc = self.test_accuracies[-1] if self.test_accuracies else 0.0
        best_test_acc = max(self.test_accuracies) if self.test_accuracies else 0.0
        print(f"\nFINAL RESULTS for {plot_suffix.upper()}:")
        print(f"Final Test Accuracy: {final_test_acc:.2%}")
        print(f"Best Test Accuracy: {best_test_acc:.2%}")

        if self.args.pos_embed == 'string':
            print(f"STRING-{self.args.string_type.upper()} implementation complete!")
            print("Check if this beats the current best method (Relative: ~90.57% on CIFAR10)")
