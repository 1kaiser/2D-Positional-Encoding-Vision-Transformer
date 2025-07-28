# filename: main.py

import os
import jax
import argparse
import datetime
from solver import Solver
from utils import print_args


def main(args):
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    solver = Solver(args)
    solver.train()
    solver.plot_graphs()


def update_args(args):
    """Update arguments, ensuring paths are absolute for checkpointing."""
    args.model_path = os.path.abspath(os.path.join(args.model_path, args.dataset))
    args.output_path = os.path.abspath(os.path.join(args.output_path, args.dataset))
    args.n_patches = (args.image_size // args.patch_size) ** 2

    if not hasattr(args, 'seed') or args.seed is None:
        args.seed = 0
    if not hasattr(args, 'dropout') or args.dropout is None:
        args.dropout = 0.1

    if args.pos_embed == 'string3d':
        args.use_depth = True
    else:
        args.use_depth = False

    print(f"Using JAX backend. Device: {jax.local_devices()}")
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D/3D Positional Embeddings for Vision Transformer (JAX)')

    # Positional Embedding
    parser.add_argument('--pos_embed', type=str, default='learn',
                       help='Type of Positional Embedding to Use in ViT',
                       choices=['none', 'learn', 'sinusoidal', 'relative', 'rope', 'string', 'string3d', 'uniform_rope'])
    
    parser.add_argument('--max_relative_distance', type=int, default=2,
                       help='max relative distance for relative PE')

    # STRING-specific arguments
    parser.add_argument('--string_type', type=str, default='cayley',
                       help='Type of STRING implementation',
                       choices=['cayley', 'circulant'])

    # 3D STRING specific arguments
    parser.add_argument('--use_depth', action='store_true',
                       help='Enable depth processing for 3D STRING (auto-set by string3d)')
    parser.add_argument('--depth_simulation', action='store_true',
                       help='Simulate depth maps for datasets without depth')
    parser.add_argument('--depth_noise_std', type=float, default=0.1,
                       help='Noise std for simulated depth maps')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs to warmup learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')
    parser.add_argument('--lr', type=float, default=5e-4, help='peak learning rate')
    parser.add_argument('--output_path', type=str, default='./outputs', help='path to store graphs')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument("--image_size", type=int, default=32, help='image size')
    parser.add_argument("--patch_size", type=int, default=4, help='patch Size')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to store dataset')

    # Model Arguments
    parser.add_argument('--model_path', type=str, default='./model', help='path to store trained model')
    parser.add_argument("--load_model", type=bool, default=False, help="load saved model")

    # JAX-specific arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed for JAX')

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args = update_args(args)
    print_args(args)

    # Print method-specific information
    if args.pos_embed == 'string':
        print(f"\nUsing 2D STRING positional encoding with {args.string_type.upper()} variant.")
    elif args.pos_embed == 'uniform_rope':
        print(f"\nUsing Uniform N-dimensional RoPE positional encoding.")
        print("This implementation follows the method proposed by Xiong et al., using uniformly spaced direction vectors.")
    elif args.pos_embed == 'string3d':
        print(f"\nUsing 3D STRING positional encoding with {args.string_type.upper()} variant.")
        if args.depth_simulation:
            print("Using simulated depth maps for CIFAR datasets.")
        print("This implementation follows the STRING robotics approach for 3D vision tasks.")

    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
