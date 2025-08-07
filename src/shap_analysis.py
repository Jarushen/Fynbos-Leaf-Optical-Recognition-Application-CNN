"""Utilities for generating SHAP explanations for image predictions.

This module uses the ``shap`` library to compute gradient‑based explanations
for convolutional neural networks.  The typical workflow involves

1. Preprocessing the input image in the same way as during training.
2. Choosing a background dataset (often a small random subset of training
   images or even a black image) against which to explain the prediction.
3. Calling :func:`compute_shap_values` to obtain the SHAP values.
4. Visualising the SHAP values as a heatmap overlay on the original image
   via :func:`shap_overlay`.

Note that computing SHAP values can be computationally expensive.  For
interactive use, keep the number of samples low (e.g., ``nsamples=50``).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
try:
    import shap  # type: ignore
except ImportError as e:
    shap = None  # type: ignore[assignment]
    _import_error = e
from typing import Optional, Tuple


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess an image for ResNet50.

    Keras' ResNet50 expects inputs preprocessed using its own
    ``preprocess_input`` function.  This helper converts the image to a
    4D batch and applies the preprocessing.

    Parameters
    ----------
    image : ndarray
        Image array with pixel values in [0, 255] and shape (H, W, 3).

    Returns
    -------
    ndarray
        Preprocessed image with shape (1, H, W, 3).
    """
    image_batch = np.expand_dims(image, axis=0).astype(np.float32)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(image_batch)
    return preprocessed


def compute_shap_values(
    model: tf.keras.Model,
    image: np.ndarray,
    background: Optional[np.ndarray] = None,
    nsamples: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for a single image.

    Parameters
    ----------
    model : tf.keras.Model
        The trained Keras model.  It should accept preprocessed images and
        output class probabilities.
    image : ndarray
        Input image array with shape (H, W, 3) and dtype uint8 or float.
    background : Optional[ndarray], default None
        Optional background dataset for the explainer.  If ``None`` a single
        black image is used.  The background must be preprocessed in the
        same way as the input (i.e., calling :func:`preprocess_image` on
        each background sample).  Shape should be (N, H, W, 3).
    nsamples : int, default 50
        Number of samples to draw when estimating SHAP values.  Smaller
        numbers result in faster but less precise explanations.

    Returns
    -------
    shap_values : ndarray
        Array of shape (1, H, W, 3) containing the SHAP values for each
        channel.  Only the SHAP values corresponding to the predicted class
        are returned.
    expected_value : ndarray
        Expected output of the model on the background dataset.  This is
        provided for completeness and is not used directly in the overlay
        visualisation.
    """
    if shap is None:
        raise RuntimeError(
            "The 'shap' library is required for SHAP analysis but is not installed. "
            "Please install it by running `pip install shap` in your environment."
        )

    # Preprocess the input image
    preprocessed_image = preprocess_image(image)

    # Determine the background dataset
    if background is None:
        # A single black image suffices as a baseline for simple tasks
        background = np.zeros_like(preprocessed_image)
    else:
        # Ensure background is a batch and preprocessed
        if background.ndim == 3:
            background = preprocess_image(background)
        elif background.ndim == 4:
            # Assume the background has already been preprocessed
            pass
        else:
            raise ValueError("Background must have 3 or 4 dimensions")

    # Build a GradientExplainer.  For classification we explain the logit
    # outputs rather than probabilities to avoid saturating softmax.
    explainer = shap.GradientExplainer(
        lambda x: model(x, training=False),
        background,
    )
    # Compute SHAP values for the image.  shap_values is a list of arrays
    # (one per output class); we select the explanation for the predicted
    # class.
    shap_values_full = explainer.shap_values(preprocessed_image, nsamples=nsamples)
    # Determine the predicted class
    preds = model.predict(preprocessed_image)
    pred_class = int(np.argmax(preds[0]))
    shap_values = shap_values_full[pred_class]
    expected_value = explainer.expected_values[pred_class]
    return shap_values, expected_value


def shap_overlay(
    image: np.ndarray,
    shap_values: np.ndarray,
    alpha: float = 0.6,
    cmap: str = "coolwarm",
) -> plt.Figure:
    """Create a matplotlib figure overlaying SHAP values onto an image.

    Parameters
    ----------
    image : ndarray
        Original RGB image with pixel values in [0, 255].
    shap_values : ndarray
        SHAP values corresponding to the image, as returned by
        :func:`compute_shap_values`.  Shape must match ``image``.
    alpha : float, default 0.6
        Transparency of the heatmap overlay (0 is fully transparent, 1
        completely hides the underlying image).
    cmap : str, default 'coolwarm'
        Matplotlib colormap to use for the SHAP heatmap.

    Returns
    -------
    figure : matplotlib.figure.Figure
        A figure containing the original image and the SHAP overlay side by
        side.
    """
    # Normalize image to [0, 1]
    img_norm = image.astype(np.float32) / 255.0
    # Aggregate SHAP values across channels by taking the mean
    shap_sum = np.mean(shap_values, axis=-1)
    # Normalise shap values for display
    vmax = np.percentile(np.abs(shap_sum), 99)
    vmin = -vmax
    shap_norm = np.clip(shap_sum, vmin, vmax)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_norm)
    axes[0].axis("off")
    axes[0].set_title("Original Image")
    # Overlay heatmap on original image
    axes[1].imshow(img_norm)
    heat = axes[1].imshow(
        shap_norm,
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
        extent=(0, img_norm.shape[1], img_norm.shape[0], 0),
    )
    axes[1].axis("off")
    axes[1].set_title("SHAP Explanation")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig