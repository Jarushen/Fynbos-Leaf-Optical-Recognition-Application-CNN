"""Top‑level package for the fynbos species classifier.

This package exposes convenience functions for loading data, creating the
transfer‑learning model, training, evaluation and SHAP analysis.  See the
individual modules for detailed documentation.
"""

__all__ = [
    "data_loader",
    "model",
    "train",
    "evaluate",
    "shap_analysis",
    "test_image",
]