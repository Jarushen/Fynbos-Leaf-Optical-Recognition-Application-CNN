"""Basic image quality assessment for fynbos images.

This module provides simple heuristics to decide whether a newly supplied
image is of sufficient quality to be added to the dataset.  It computes
several statistics (resolution, sharpness and brightness) and returns a
boolean indicating whether the image passes all checks.  You can adjust
thresholds to suit your own requirements.

The assessment criteria are:

* **Resolution** – The smallest dimension (height or width) must be at
  least ``MIN_DIM`` pixels.  Low resolution images may lack detail needed
  for classification.
* **Sharpness** – Measured by the variance of the Laplacian.  A blurred
  image yields a low variance.  Images with sharpness below
  ``MIN_SHARPNESS`` are rejected.
* **Brightness** – The mean pixel intensity (0–255) should lie within
  ``BRIGHTNESS_RANGE``.  Very dark or very bright images may obscure
  relevant features.

If all three criteria are met the image is considered acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore[assignment]
import numpy as np
from PIL import Image


@dataclass
class QualityCriteria:
    """Thresholds for image quality assessment."""
    min_dim: int = 224
    min_sharpness: float = 100.0
    brightness_range: Tuple[float, float] = (50.0, 200.0)


def compute_sharpness(image: np.ndarray) -> float:
    """Compute a simple focus measure based on the variance of the Laplacian.

    A high variance indicates a sharp image whereas a low value suggests
    blurriness.  Images must be converted to grayscale for this measure.

    Parameters
    ----------
    image : ndarray
        RGB image as a NumPy array.

    Returns
    -------
    float
        Variance of the Laplacian of the grayscale image.
    """
    if cv2 is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)
    else:
        # Fallback: approximate sharpness using gradient magnitude variance
        # Convert to grayscale manually
        gray = np.mean(image, axis=2)
        # Compute gradients along x and y
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        # Pad to original shape
        dx = np.pad(dx, ((0, 0), (0, 1)), mode='constant')
        dy = np.pad(dy, ((0, 1), (0, 0)), mode='constant')
        grad_mag = np.sqrt(dx**2 + dy**2)
        return float(grad_mag.var())


def compute_brightness(image: np.ndarray) -> float:
    """Compute the mean brightness of an image.

    Brightness is the average of pixel values in grayscale.  Values are in
    the range [0, 255].

    Parameters
    ----------
    image : ndarray
        RGB image as a NumPy array.

    Returns
    -------
    float
        Mean brightness.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())


def assess_image_quality(
    image_path: str,
    criteria: QualityCriteria | None = None,
    verbose: bool = False,
) -> bool:
    """Determine whether an image meets minimum quality requirements.

    Parameters
    ----------
    image_path : str
        Path to the image file to assess.
    criteria : QualityCriteria, optional
        Thresholds for the checks.  If ``None`` defaults are used.
    verbose : bool, default False
        Whether to print detailed scores and decisions.

    Returns
    -------
    bool
        ``True`` if the image passes all checks, ``False`` otherwise.
    """
    if criteria is None:
        criteria = QualityCriteria()

    # Load image via Pillow and convert to RGB numpy array
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape

    # Resolution check
    min_dim = min(h, w)
    resolution_ok = min_dim >= criteria.min_dim
    # Sharpness check
    sharpness = compute_sharpness(arr)
    sharpness_ok = sharpness >= criteria.min_sharpness
    # Brightness check
    brightness = compute_brightness(arr)
    bright_ok = criteria.brightness_range[0] <= brightness <= criteria.brightness_range[1]

    if verbose:
        print(f"Image size: {w}×{h} (min dimension {min_dim} px) – {'OK' if resolution_ok else 'Too small'}")
        print(f"Sharpness: {sharpness:.2f} – {'OK' if sharpness_ok else 'Too blurry'}")
        print(f"Brightness: {brightness:.2f} – {'OK' if bright_ok else 'Too dark/bright'}")

    return resolution_ok and sharpness_ok and bright_ok