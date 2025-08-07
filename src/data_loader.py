"""Utilities for loading and preprocessing image datasets.

This module defines functions to load training, validation and test datasets
from a folder structure on disk.  It uses Keras' high‑level API to create
`tf.data.Dataset` objects with batching, shuffling and optional data
augmentation.  The loader infers class names from subdirectory names.

Example directory layout::

    data/
      train/
        class1/
        class2/
        ...
      val/
        class1/
        class2/
        ...
      test/
        class1/
        class2/
        ...

Functions
---------
load_datasets
    Loads the training, validation and test datasets from the given
    directory.
"""

from __future__ import annotations

import os
from typing import Tuple, Optional

import tensorflow as tf


def load_dataset_from_dir(
    directory: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 123
) -> Tuple[tf.data.Dataset, list[str]]:
    """Load an image dataset from a directory.

    The function uses :func:`tf.keras.preprocessing.image_dataset_from_directory`
    to load images and labels.  It automatically infers the class names from
    subdirectory names under ``directory``.

    Parameters
    ----------
    directory : str
        Path to a folder containing one subdirectory per class.
    img_size : tuple of int, default (224, 224)
        Target size to which images are resized.  A 224×224 square is a good
        default for ResNet50.
    batch_size : int, default 32
        Number of samples per batch.
    shuffle : bool, default True
        Whether to shuffle the dataset.  Set to ``False`` for deterministic
        ordering (e.g., evaluation).
    seed : int, default 123
        Random seed used for shuffling and transformations.

    Returns
    -------
    dataset : tf.data.Dataset
        A batched dataset of (image, label) pairs.
    class_names : list of str
        A list of class names inferred from subdirectory names.

    Notes
    -----
    Images are loaded in RGB mode and converted to floating point tensors in
    the range [0, 1].
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    class_names = dataset.class_names
    # Normalize images to [0, 1]
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    return dataset, class_names


def load_datasets(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 123,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load training, validation and test datasets.

    Given a root ``data_dir`` this function looks for ``train``, ``val``
    and ``test`` subdirectories and loads each as a separate dataset.  All
    datasets share the same class names.

    Parameters
    ----------
    data_dir : str
        Path to a directory containing ``train``, ``val`` and ``test``.
    img_size : tuple of int, default (224, 224)
        Image size to which all inputs are resized.
    batch_size : int, default 32
        Batch size for the datasets.
    seed : int, default 123
        Random seed for deterministic splits and augmentations.

    Returns
    -------
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    test_ds : tf.data.Dataset
        Test dataset.
    class_names : list of str
        List of class labels shared across all datasets.

    Raises
    ------
    FileNotFoundError
        If any of the required subdirectories is missing.
    """
    required_subdirs = ["train", "val", "test"]
    for sub in required_subdirs:
        path = os.path.join(data_dir, sub)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Expected subdirectory '{sub}' under {data_dir}")

    train_ds, class_names = load_dataset_from_dir(
        os.path.join(data_dir, "train"),
        img_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds, _ = load_dataset_from_dir(
        os.path.join(data_dir, "val"),
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_ds, _ = load_dataset_from_dir(
        os.path.join(data_dir, "test"),
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    return train_ds, val_ds, test_ds, class_names