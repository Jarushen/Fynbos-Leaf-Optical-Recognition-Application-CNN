# Fynbos Species Classifier

This repository provides a complete workflow for building, training and deploying
a convolutional neural network (CNN) to identify fynbos species from images.

Fynbos is a unique vegetation type found primarily in the Western Cape of
South Africa. Accurate species identification can be challenging, so this
project uses a transfer‐learning approach based on a pre–trained ResNet50
architecture to recognise species in photographs.   A simple test
environment is included for quickly classifying new images and visualising
SHAP explanations.

The dataset used to train this model was manually created and can be accessed here:

https://drive.google.com/drive/folders/11EpB1b3q7X9voYaSR5ghsv0J5E7pxhhQ?usp=sharing

Please contact Jarushen Govender at govenderjarushen@gmail.com for any queries on the dataset.




## Repository structure

```
fynbos_cnn/
│
├── notebooks/             – Example notebooks and interactive test environment
│   └── test_environment.ipynb
│
├── src/                   – Core source code
│   ├── __init__.py
│   ├── data_loader.py     – Dataset loading and preprocessing
│   ├── model.py           – Definition of the transfer learning model
│   ├── train.py           – Command‑line script for training
│   ├── evaluate.py        – Script for evaluating a trained model
│   ├── shap_analysis.py   – Utilities for computing SHAP explanations
│   ├── image_quality.py   – Simple image quality assessment tools
│   └── test_image.py      – Script for classifying a single image and producing SHAP output
│
├── requirements.txt       – Python package dependencies
└── README.md              – This file
```

### Notebooks

The `notebooks/test_environment.ipynb` notebook provides an interactive way to
upload an image, run it through a trained model and view both the predicted
class (with probabilities) and a SHAP explanation.  The notebook uses
`ipywidgets` for file upload and displays the original image alongside the
SHAP heatmap.

### Source code

The source code under `src/` is organised into small, composable modules:

* **`data_loader.py`**: Defines functions to load training, validation and test
  datasets from a directory structure.  Data augmentation and resizing is
  applied via Keras utilities.  The code expects a folder structure where
  each class is contained in its own subdirectory.  For example:

  ```
  data/
    train/
      species1/
      species2/
      ...
    val/
      species1/
      species2/
      ...
    test/
      species1/
      species2/
      ...
  ```

* **`model.py`**: Constructs a transfer‑learning model based on ResNet50
  pre‑trained on ImageNet.  The base layers are optionally frozen and a small
  classification head is appended to match the number of classes in the
  dataset.

* **`train.py`**: Provides a command‑line interface for training the model.
  It accepts arguments such as dataset directory, batch size, number of
  epochs and output path.  The trained model is saved in TensorFlow’s
  `SavedModel` format by default.  Training history plots are also saved
  automatically.

* **`evaluate.py`**: Loads a saved model and evaluates it on a test dataset,
  reporting overall accuracy and per‑class metrics.

* **`shap_analysis.py`**: Contains functions to compute SHAP values for a
  given image and model.  It uses the `shap` library’s `GradientExplainer` to
  generate a heatmap that highlights which pixels most influence the
  prediction.

* **`image_quality.py`**: Provides a basic heuristic to decide whether new
  images are of sufficient quality to be added to the training dataset.  It
  checks the minimum resolution, sharpness (via the variance of the Laplacian)
  and brightness.  Use this to validate images before uploading them.

* **`test_image.py`**: A standalone script that accepts a path to an image
  and a saved model, then prints the predicted class, the associated
  probabilities and writes a SHAP explanation image to disk.  This script
  underpins the notebook test environment and can also be used as a CLI tool.

## Installation

1. Clone this repository or download the source code.
2. Create a Python virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Data

The Fynbos image dataset is not bundled with this repository due to its size.
You will need to download the images from the Google Drive link provided by
the project author and organise them into `train`, `val` and `test`
directories as shown above.  The number of classes is inferred automatically
from the subdirectory names.

## Training the model

To train the model from scratch on your fynbos dataset run:

```bash
python src/train.py \
  --data-dir /path/to/data \
  --output-dir /path/to/save/model \
  --epochs 20 \
  --batch-size 32
```

This script will create a timestamped folder in the specified output
directory containing the saved model and training history plots.  You can
adjust hyperparameters via command‑line arguments; run

```bash
python src/train.py --help
```

for a complete list.

## Evaluating a trained model

After training, evaluate the model on your test set using:

```bash
python src/evaluate.py \
  --data-dir /path/to/data/test \
  --model-dir /path/to/save/model
```

This will output overall accuracy as well as a confusion matrix.

## Classifying a single image

Use the `test_image.py` script to classify a new image and generate a SHAP
explanation:

```bash
python src/test_image.py \
  --image /path/to/image.jpg \
  --model-dir /path/to/save/model \
  --output shap_output.png
```

The script prints the predicted class and class probabilities to the console
and writes the SHAP heatmap overlay to the specified output file.

## Interactive testing with Jupyter

If you prefer a more interactive experience, open the `notebooks/test_environment.ipynb`
notebook in JupyterLab or VS Code.  The notebook uses `ipywidgets` to let you
upload an image from your local machine and see the prediction along with
its SHAP explanation directly in the notebook.

## Coding standards

The codebase follows modern Python best practices:

* Modules are short and focused on a single responsibility.
* Type annotations are used throughout to clarify expected argument and
  return types.
* Docstrings follow the [Google Python style](https://google.github.io/styleguide/pyguide.html) with
  concise summaries and parameter descriptions.
* Logging is used instead of print statements for better control over
  verbosity.

Feel free to adapt and extend this repository according to your needs.  If
you spot an issue or have suggestions for improvement, please open an
issue or submit a pull request.
