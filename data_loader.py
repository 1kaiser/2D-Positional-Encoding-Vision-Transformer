# filename: data_loader.py

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import jax
import jax.numpy as jnp

def simulate_depth_map(image, noise_std=0.1):
    """
    Simulate depth map from RGB image for datasets without depth.
    Uses image gradients and intensity to create plausible depth estimates.
    """
    gray = tf.reduce_mean(image, axis=-1, keepdims=True)
    gray_expanded = tf.expand_dims(gray, 0)
    grad_x = tf.image.sobel_edges(gray_expanded)[0, :, :, 0, 0:1]
    grad_y = tf.image.sobel_edges(gray_expanded)[0, :, :, 0, 1:2]
    grad_mag = tf.sqrt(grad_x**2 + grad_y**2)
    
    intensity_depth = 1.0 - gray
    gradient_depth = tf.nn.sigmoid(grad_mag) * 0.3
    depth_base = intensity_depth + gradient_depth
    depth_normalized = tf.nn.sigmoid(depth_base - 0.5)

    h, w = tf.shape(depth_normalized)[0], tf.shape(depth_normalized)[1]
    y_coords = tf.range(h, dtype=tf.float32) / tf.cast(h - 1, tf.float32)
    x_coords = tf.range(w, dtype=tf.float32) / tf.cast(w - 1, tf.float32)
    y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing='ij')
    dist_from_center = tf.sqrt((x_grid - 0.5)**2 + (y_grid - 0.5)**2)
    spatial_bias = tf.expand_dims(tf.nn.sigmoid(2.0 - 4.0 * dist_from_center), -1)
    
    depth_map = depth_normalized * 0.7 + spatial_bias * 0.3
    
    if noise_std > 0:
        noise = tf.random.normal(tf.shape(depth_map), stddev=noise_std)
        depth_map = tf.clip_by_value(depth_map + noise, 0.0, 1.0)
        
    return depth_map

def preprocess_image(image, label, image_size, num_classes, simulate_depth=False, depth_noise_std=0.1):
    """Preprocesses the image and converts label to one-hot encoding."""
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_classes)

    if simulate_depth:
        depth_map = simulate_depth_map(image, depth_noise_std)
        return image, depth_map, label
    else:
        return image, label

def create_dataset(dataset_name, split, image_size, num_classes, batch_size,
                  shuffle=True, shuffle_buffer_size=10000, simulate_depth=False, depth_noise_std=0.1):
    """Loads and preprocesses a dataset using TensorFlow Datasets."""
    ds = tfds.load(dataset_name, split=split, as_supervised=True, data_dir='./data/')

    preprocess_fn = lambda img, lbl: preprocess_image(
        img, lbl, image_size, num_classes, simulate_depth, depth_noise_std
    )
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(batch_size, drop_remainder=True) # Use drop_remainder for stable JAX shapes
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def get_loader(args):
    """Loads dataset and provides iterators for JAX, supporting depth simulation."""
    simulate_depth = getattr(args, 'depth_simulation', False) or (getattr(args, 'pos_embed', '') == 'string3d')
    depth_noise_std = getattr(args, 'depth_noise_std', 0.1)

    if simulate_depth:
        print(f"INFO: Simulating depth maps with noise std={depth_noise_std}")

    train_ds = create_dataset(
        args.dataset, 'train', args.image_size, args.n_classes, args.batch_size,
        shuffle=True, simulate_depth=simulate_depth, depth_noise_std=depth_noise_std
    )
    test_ds = create_dataset(
        args.dataset, 'test', args.image_size, args.n_classes, args.batch_size,
        shuffle=False, simulate_depth=simulate_depth, depth_noise_std=depth_noise_std
    )

    return tfds.as_numpy(train_ds), tfds.as_numpy(test_ds)

# ==============================================================================
# === THE FIX IS IN THIS FUNCTION BELOW ===
# ==============================================================================

def get_batch_data(batch, use_depth=False):
    """
    Utility function to extract data from a batch consistently.
    This version ALWAYS returns three values (images, depth_map, labels),
    with depth_map being None if it's not used.
    """
    if use_depth:
        # Expecting (images, depth_maps, labels)
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        # Fallback if depth simulation failed or wasn't triggered properly
        else:
            return batch[0], None, batch[1]
    else:
        # Expecting (images, labels)
        # We return (images, None, labels) to maintain a consistent signature.
        return batch[0], None, batch[1]
