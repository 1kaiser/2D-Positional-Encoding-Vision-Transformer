import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import jax
import jax.numpy as jnp

def preprocess_image(image, label, image_size, num_classes):
    """Preprocesses the image and converts label to one-hot encoding."""
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_classes)
    return image, label

def create_dataset(dataset_name, split, image_size, num_classes, batch_size, shuffle=True, shuffle_buffer_size=10000):
    """Loads and preprocesses a dataset using TensorFlow Datasets."""
    ds = tfds.load(dataset_name, split=split, as_supervised=True)

    ds = ds.map(lambda img, label: preprocess_image(img, label, image_size, num_classes))

    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def get_loader(args):
    """
    Loads CIFAR10 or CIFAR100 dataset using TensorFlow Datasets
    and provides iterators for JAX compatibility.
    """
    train_ds = create_dataset(
        args.dataset,
        split='train',
        image_size=args.image_size,
        num_classes=args.n_classes,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_ds = create_dataset(
        args.dataset,
        split='test',
        image_size=args.image_size,
        num_classes=args.n_classes,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Convert TensorFlow datasets to NumPy iterators for JAX compatibility
    train_loader = tfds.as_numpy(train_ds)
    test_loader = tfds.as_numpy(test_ds)

    return train_loader, test_loader

# Example usage (for testing the data loader)
# if __name__ == '__main__':
#     class Args:
#         def __init__(self):
#             self.dataset = 'cifar10'
#             self.image_size = 32
#             self.n_classes = 10
#             self.batch_size = 128

#     args = Args()
#     train_loader, test_loader = get_loader(args)

#     print("Testing train loader:")
#     for images, labels in train_loader:
#         print(f"Batch shape (images): {images.shape}, Batch shape (labels): {labels.shape}")
#         print(f"Image dtype: {images.dtype}, Label dtype: {labels.dtype}")
#         break # Process only one batch

#     print("\nTesting test loader:")
#     for images, labels in test_loader:
#         print(f"Batch shape (images): {images.shape}, Batch shape (labels): {labels.shape}")
#         print(f"Image dtype: {images.dtype}, Label dtype: {labels.dtype}")
#         break # Process only one batch
