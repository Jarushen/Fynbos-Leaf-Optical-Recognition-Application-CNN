"""Command‑line script to train the fynbos classifier.

This script loads image data from a directory, constructs a ResNet50
transfer‑learning model and trains it on the data.  Training progress is
reported and the model is saved to disk along with training history plots.

Usage
-----

From the repository root run::

    python src/train.py \
      --data-dir /path/to/data \
      --output-dir ./models \
      --epochs 20 \
      --batch-size 32

See ``python src/train.py --help`` for all available options.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import tensorflow as tf

from .data_loader import load_datasets
from .model import build_model


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Train a fynbos image classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing train/val/test subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory in which to save the trained model and plots",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Keep the ResNet base frozen (default).  If set, the base is frozen; if omitted, the base is trainable.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the Adam optimizer",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate applied before the classification layer",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement to wait before early stopping",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"Loading datasets from {data_dir}…")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir,
        img_size=(224, 224),
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Build the model
    model = build_model(
        num_classes=num_classes,
        input_shape=(224, 224, 3),
        base_trainable=not args.freeze_base,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
    )

    model.summary()

    # Callbacks: early stopping and model checkpoint
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model"
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        save_best_only=True,
    )

    # Train
    print("Starting training…")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stop, checkpoint_cb],
    )

    # Save final model (best weights are already restored)
    final_model_path = run_dir / "saved_model"
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Persist class names to a text file for later inference
    class_file = run_dir / "class_names.txt"
    with class_file.open("w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Class names saved to {class_file}")

    # Plot and save training curves
    history_fig_path = run_dir / "training_history.png"
    plt.figure(figsize=(10, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig(history_fig_path)
    plt.close()
    print(f"Training history saved to {history_fig_path}")


if __name__ == "__main__":
    main()