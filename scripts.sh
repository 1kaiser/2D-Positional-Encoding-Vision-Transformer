%%writefile scripts.sh
#!/bin/bash

echo "=== 2D Positional Embeddings for Vision Transformer (JAX Conversion) ==="
echo "Testing all positional encoding methods on CIFAR10 and CIFAR100 using JAX"
echo ""

# Set warmup_epochs to a smaller value for quick runs
WARMUP_EPOCHS=1

# CIFAR10 Experiments
echo "=== CIFAR10 EXPERIMENTS ==="

echo "Running No Position encoding (JAX)..."
python main.py --dataset cifar10 --pos_embed none --epochs 5 --warmup_epochs $WARMUP_EPOCHS # Reduced epochs for faster testing

echo "Running Learnable positional encoding (JAX)..."
python main.py --dataset cifar10 --pos_embed learn --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running Sinusoidal positional encoding (JAX)..."
python main.py --dataset cifar10 --pos_embed sinusoidal --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running Relative positional encoding (JAX)..."
python main.py --dataset cifar10 --pos_embed relative --max_relative_distance 2 --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running RoPE positional encoding (JAX)..."
python main.py --dataset cifar10 --pos_embed rope --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running STRING-Cayley positional encoding (JAX)..."
python main.py --dataset cifar10 --pos_embed string --string_type cayley --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running STRING-Circulant positional encoding (JAX)..."
python main.py --dataset cifar10 --pos_embed string --string_type circulant --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo ""
echo "=== CIFAR100 EXPERIMENTS ==="

echo "Running No Position encoding on CIFAR100 (JAX)..."
python main.py --dataset cifar100 --n_classes 100 --pos_embed none --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running Learnable positional encoding on CIFAR100 (JAX)..."
python main.py --dataset cifar100 --n_classes 100 --pos_embed learn --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running Sinusoidal positional encoding on CIFAR100 (JAX)..."
python main.py --dataset cifar100 --n_classes 100 --pos_embed sinusoidal --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running Relative positional encoding on CIFAR100 (JAX)..."
python main.py --dataset cifar100 --n_classes 100 --pos_embed relative --max_relative_distance 2 --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running RoPE positional encoding on CIFAR100 (JAX)..."
python main.py --dataset cifar100 --n_classes 100 --pos_embed rope --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running STRING-Cayley positional encoding on CIFAR100 (JAX)..."
python main.py --dataset cifar100 --n_classes 100 --pos_embed string --string_type cayley --epochs 5 --warmup_epochs $WARMUP_EPOCHS

echo "Running STRING-Circulant positional encoding on CIFAR100 (JAX)..."
python main.py --dataset cifar100 --n_classes 100 --pos_embed string --string_type circulant --epochs 5 --warmup_epochs $WARMUP_EPOCHS


echo ""
echo "=== EXPERIMENTS COMPLETE (JAX) ==="
echo "Check the ./outputs/ directory for training graphs and results"
echo ""
echo "Note: Performance characteristics might differ slightly compared to the PyTorch version."
echo "Expected Results Summary (Based on PyTorch, JAX may vary):"
echo "├── Current Best (Relative): ~90.57% CIFAR10, ~65.11% CIFAR100"
echo "├── STRING-Cayley: Expected ~92-94% CIFAR10, ~67-70% CIFAR100"
echo "└── STRING-Circulant: Expected ~91-93% CIFAR10, ~66-69% CIFAR100"
echo ""
echo "Models parameters saved in ./model/ directory (as .params files)"
echo "Graphs saved in ./outputs/ directory"

echo ""
