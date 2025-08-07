"""Model definition for the fynbos classifier.

This module defines a function to build a convolutional neural network based on
ResNet50 using transfer learning.  The base network is loaded with ImageNet
weights and can be optionally frozen.  A lightweight classification head is
added to adapt to the number of target classes.

Functions
---------
build_model
    Construct a compiled Keras model ready for training.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

def build_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    base_trainable: bool = False,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Construct and compile the transfer‑learning model.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shape : tuple of int, default (224, 224, 3)
        Shape of input images (height, width, channels).
    base_trainable : bool, default False
        Whether to unfreeze the base ResNet50 layers for fine‑tuning.
    dropout_rate : float, default 0.5
        Dropout rate applied before the final classification layer.
    learning_rate : float, default 1e-4
        Learning rate for the Adam optimizer.

    Returns
    -------
    model : tf.keras.Model
        A compiled Keras model ready to be trained.
    """
    # Load the ResNet50 base model without the top classifier layers.
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = base_trainable

    inputs = tf.keras.Input(shape=input_shape)
    # Preprocess inputs according to ResNet50 expectations.
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile the model with a suitable optimizer and loss.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model