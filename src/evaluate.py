"""Evaluate a trained fynbos classifier on a test dataset.

This script loads a saved TensorFlow model and computes the classification
accuracy on a test dataset.  Optionally, it prints a confusion matrix and
per‑class precision/recall/f1 using scikit‑learn.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

from .data_loader import load_dataset_from_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fynbos classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing test images organised by class",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the saved model (as produced by train.py)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load the model
    model_path = Path(args.model_dir)
    model = tf.keras.models.load_model(model_path)
    # Load test dataset
    test_ds, class_names = load_dataset_from_dir(
        args.data_dir,
        img_size=(224, 224),
        batch_size=32,
        shuffle=False,
    )
    # Collect predictions and labels
    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[np.ndarray] = []
    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images)
        y_scores.extend(preds)
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    accuracy = np.mean(y_true_arr == y_pred_arr)
    print(f"Test accuracy: {accuracy*100:.2f}%")

    # Print classification report
    report = classification_report(y_true_arr, y_pred_arr, target_names=class_names)
    print("\nClassification report:\n")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    print("Confusion matrix:")
    print(tabulate(cm, headers=class_names, showindex=class_names, tablefmt="grid"))


if __name__ == "__main__":
    main()