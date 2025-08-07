"""Classify a single image and generate a SHAP explanation.

This script loads a trained model and a list of class names, then reads
an image file from disk, predicts its class probabilities and writes a
SHAP heatmap overlay to an output image.  It is intended as a quick way
to test the classifier on new images.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import tensorflow as tf

from .shap_analysis import compute_shap_values, shap_overlay


def load_class_names(model_dir: Path) -> List[str]:
    """Load class names from a text file saved during training.

    The training script writes a ``class_names.txt`` file into the run
    directory.  This helper reads that file and returns the classes in
    order.  If the file is not found, an empty list is returned.

    Parameters
    ----------
    model_dir : Path
        Path to the directory containing the saved model and
        ``class_names.txt``.

    Returns
    -------
    list of str
        List of class names, or an empty list if not found.
    """
    class_file = model_dir / "class_names.txt"
    if not class_file.exists():
        print(f"Warning: class_names.txt not found in {model_dir}. Predictions will be indices.")
        return []
    classes: List[str] = []
    with class_file.open("r", encoding="utf-8") as f:
        for line in f:
            classes.append(line.strip())
    return classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify an image and output SHAP explanation")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image to classify",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the saved model (with class_names.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="shap_output.png",
        help="Path where the SHAP overlay image will be saved",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=50,
        help="Number of samples used by SHAP for explanation (smaller = faster)",
    )
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help=(
            "Skip the image quality assessment. By default the image is "
            "checked for minimum resolution, sharpness and brightness before classification."
        ),
    )
    return parser.parse_args()


def load_image(image_path: Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load an image from disk and resize it.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    target_size : tuple of int, default (224, 224)
        Target spatial dimensions.

    Returns
    -------
    ndarray
        The resized image as a NumPy array in RGB format with dtype uint8.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    return np.array(img)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)

    # Optionally run image quality assessment
    if not args.no_quality_check:
        try:
            from .image_quality import assess_image_quality
        except Exception as e:
            print(
                "Quality assessment module could not be loaded. Proceeding without quality check."
            )
        else:
            ok = assess_image_quality(str(image_path), verbose=True)
            if not ok:
                print(
                    "Image did not meet quality thresholds and will not be classified. "
                    "Use --no-quality-check to override."
                )
                return

    # Load model
    model = tf.keras.models.load_model(model_dir / "saved_model")
    # Load class names
    class_names = load_class_names(model_dir)

    # Load and preprocess image
    image = load_image(image_path)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(
        np.expand_dims(image, axis=0).astype(np.float32)
    )

    # Predict
    preds = model.predict(preprocessed)
    pred_probs = preds[0]
    pred_index = int(np.argmax(pred_probs))
    pred_class_name = (
        class_names[pred_index] if class_names and pred_index < len(class_names) else str(pred_index)
    )
    # Print results
    print("Prediction results")
    print("==================")
    print(f"Predicted class: {pred_class_name} (index {pred_index})")
    print("Class probabilities:")
    for idx, prob in enumerate(pred_probs):
        name = class_names[idx] if class_names and idx < len(class_names) else str(idx)
        print(f"  {name}: {prob*100:.2f}%")

    # Compute SHAP values
    shap_values, expected_value = compute_shap_values(
        model,
        image=image,
        background=None,
        nsamples=args.nsamples,
    )
    # Create overlay figure
    fig = shap_overlay(image, shap_values)
    # Save figure
    fig.savefig(output_path)
    print(f"SHAP explanation saved to {output_path}")


if __name__ == "__main__":
    main()