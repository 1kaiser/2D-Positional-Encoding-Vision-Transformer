import os
import jax
import argparse
import datetime
from solver import Solver # Import the JAX Solver
from utils import print_args # Import the JAX-compatible print_args


def main(args):
    # Create required directories if they don't exist
    os.makedirs(args.model_path,  exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    solver = Solver(args)
    solver.train()               # Training function
    solver.plot_graphs()         # Training plots
    # Note: The JAX Solver's test method is called within train after each epoch.
    # A final test can be explicitly called if needed, but the plotting includes test results.
    # solver.test(train=True) # If you want a separate final test run

# Update arguments
def update_args(args):
    args.model_path  = os.path.join(args.model_path, args.dataset)
    args.output_path = os.path.join(args.output_path, args.dataset)
    args.n_patches   = (args.image_size // args.patch_size) ** 2
    # JAX handles device placement more automatically.
    # We can check device availability but don't need to set args.is_cuda
    # as it's not used by the JAX Solver.
    # args.is_cuda = torch.cuda.is_available() # No longer needed

    # Add a seed argument for JAX randomness
    if not hasattr(args, 'seed') or args.seed is None:
        args.seed = 0 # Default seed if not provided

    # Add dropout argument if it doesn't exist (used by Flax model)
    if not hasattr(args, 'dropout') or args.dropout is None:
        args.dropout = 0.1 # Default dropout rate

    print(f"Using JAX backend. Device: {jax.local_devices()}")

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Positional Embeddings for Vision Transformer (JAX)')

    # Positional Embedding
    parser.add_argument('--pos_embed', type=str, default='learn',
                       help='Type of Positional Embedding to Use in ViT',
                       choices=['none', 'learn', 'sinusoidal', 'relative', 'rope', 'string'])

    parser.add_argument('--max_relative_distance', type=int, default=2,
                       help='max relative distance used only in relative type positional embedding (referred to as k in paper)')

    # STRING-specific arguments
    parser.add_argument('--string_type', type=str, default='cayley',
                       help='Type of STRING implementation to use',
                       choices=['cayley', 'circulant'])

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs to warmup learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes in the dataset')
    # n_workers is handled by tf.data, setting it here won't change tf.data behavior directly
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders (used by tf.data)')
    parser.add_argument('--lr', type=float, default=5e-4, help='peak learning rate')
    parser.add_argument('--output_path', type=str, default='./outputs', help='path to store training graphs and tsne plots')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate') # Add dropout arg

    # Data arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument("--image_size", type=int, default=32, help='image size')
    parser.add_argument("--patch_size", type=int, default=4, help='patch Size')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to store downloaded dataset') # tfds handles data path

    # Model Arguments
    parser.add_argument('--model_path', type=str, default='./model', help='path to store trained model')
    parser.add_argument("--load_model", type=bool, default=False, help="load saved model")

    # Add JAX-specific arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed for JAX')


    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args = update_args(args)
    print_args(args)

    # Print STRING-specific info if using STRING
    if args.pos_embed == 'string':
        print(f"Using STRING positional encoding with {args.string_type.upper()} variant")
        print("Expected improvements over standard methods!")

    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
