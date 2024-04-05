# FLORA-CNN Project Overview

The Fynbos Leaf Optical Recognition Application (or FLORA) is a novel machine- learning tool created for the purposes of aiding conservation efforts of the Cape Floral Region, and the species of plants known as Fynbos in particular. Known for their distinctive evolutionary features, the species maintains a revered position in the ecological heritage of South Africa.

This version of FLORA intends to make use of a Convolutional Neural Network trained on a dataset of collected leaf images, to correctly classify species of Fynbos using natural images as training data. The thesis intends to combat many of the pitfalls of using CNN technology such as working with small datasets and provides a novel approach for dealing with image quality issues and over-fitting that arise from working with limited data. This project also intends to be scalable, and to be able to grow and become more generalised as more training data are added.

The collected data involved manual sample collection using photography equipment and consists of 1,196 field images spread across 35 different species of plants. A part of thesis involved the creation of a novel Image Quality Assessment tool to remove low quality images that negatively influenced the predictive capability of the model.

The model evaluation process makes use of SHapley Additive exPlanations (SHAP), a tool for visualising model predictions, to contribute to the explian-ability of the model and to develop trust and confidence in machine-learning algorithms, with the ultimate aim of providing a tool to merge the fields of ecology, botany and electrical engineering.

Multiple models were trained and evaluated and the selected model for the project obtained a classification accuracy of 76% on the validation data, and an F1-score of 74%. This was an extremely positive result as the training data consisted of exclusively natural images and no feature engineering was performed. The model was then tuned to specific hyper-parameter values which yielded a small performance boost, and then tested on its ability to generalise. Five new classes were added to the training set and the model performance remained consistent, demonstrating robust generalisation. This project contributes knowledge to the growing field of image recognition, and provides a clear framework for model explain-ability which should benefit future research endeavors.

# Requirements
This project is written in Python to make use of several machine learning libraries like Tensorflow in order to work. 

# Dataset
The dataset consists of roughly 1200 images which I collected from Kirstenbosch Botanical Garden using a variety of different cameras and devices over a period of 2 weeks during July 2021. There are 56 species of plants, not all of which are Fynbos.

# Model 1

# Model 2

# Model 3 

# Image Quality Assessment

# SHAP Analysis
