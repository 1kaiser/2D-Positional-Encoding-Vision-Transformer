#!/bin/bash

# ==============================================================================
# 2D/3D Positional Embeddings for Vision Transformer (JAX) - Test Runner
#
# This script tests all available positional encoding methods on both
# CIFAR10 and CIFAR100 datasets.
#
# Usage:
#   - For a quick test (5 epochs): ./scripts.sh
#   - For a longer run (50 epochs): ./scripts.sh 50
# ==============================================================================

# --- 1. SETTINGS ---

# Check for a command-line argument for the number of epochs
if [ -z "$1" ]; then
  # If no argument is provided, default to 5 epochs
  EPOCHS=5
  echo "INFO: No epoch count provided. Defaulting to $EPOCHS epochs."
  echo "      To specify the number of epochs, run: ./scripts.sh [number_of_epochs]"
else
  # Use the provided argument as the number of epochs
  EPOCHS=$1
  echo "INFO: Running all experiments for $EPOCHS epochs."
fi

# Set a fixed number of warmup epochs for the learning rate schedule
WARMUP_EPOCHS=1
echo "INFO: Using $WARMUP_EPOCHS warmup epochs for all runs."
echo ""


# --- 2. CIFAR10 EXPERIMENTS ---

echo "==========================="
echo "=== CIFAR10 EXPERIMENTS ==="
echo "==========================="

echo "--> Running: NONE"
python main.py --dataset cifar10 --pos_embed none --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: LEARN"
python main.py --dataset cifar10 --pos_embed learn --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: SINUSOIDAL"
python main.py --dataset cifar10 --pos_embed sinusoidal --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: RELATIVE"
python main.py --dataset cifar10 --pos_embed relative --max_relative_distance 2 --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: ROPE"
python main.py --dataset cifar10 --pos_embed rope --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: UNIFORM_ROPE"
python main.py --dataset cifar10 --pos_embed uniform_rope --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: STRING_CAYLEY"
python main.py --dataset cifar10 --pos_embed string --string_type cayley --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: STRING_CIRCULANT"
python main.py --dataset cifar10 --pos_embed string --string_type circulant --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: STRING3D_CAYLEY (with depth simulation)"
python main.py --dataset cifar10 --pos_embed string3d --string_type cayley --depth_simulation --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo ""
# --- 3. CIFAR100 EXPERIMENTS ---

echo "============================="
echo "=== CIFAR100 EXPERIMENTS ==="
echo "============================="

echo "--> Running: NONE"
python main.py --dataset cifar100 --n_classes 100 --pos_embed none --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: LEARN"
python main.py --dataset cifar100 --n_classes 100 --pos_embed learn --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: SINUSOIDAL"
python main.py --dataset cifar100 --n_classes 100 --pos_embed sinusoidal --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: RELATIVE"
python main.py --dataset cifar100 --n_classes 100 --pos_embed relative --max_relative_distance 2 --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: ROPE"
python main.py --dataset cifar100 --n_classes 100 --pos_embed rope --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: UNIFORM_ROPE"
python main.py --dataset cifar100 --n_classes 100 --pos_embed uniform_rope --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: STRING_CAYLEY"
python main.py --dataset cifar100 --n_classes 100 --pos_embed string --string_type cayley --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: STRING_CIRCULANT"
python main.py --dataset cifar100 --n_classes 100 --pos_embed string --string_type circulant --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo "--> Running: STRING3D_CAYLEY (with depth simulation)"
python main.py --dataset cifar100 --n_classes 100 --pos_embed string3d --string_type cayley --depth_simulation --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS

echo ""
# --- 4. COMPLETION ---

echo "=========================="
echo "=== EXPERIMENTS COMPLETE ==="
echo "=========================="
echo "Models were trained for $EPOCHS epochs."
echo "Trained model weights are saved in the './model/' directory."
echo "Training graphs are saved in the './outputs/' directory."
echo ""
