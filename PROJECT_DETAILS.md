# Pixel Predictor

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Modules](#modules)
4. [Classes](#classes)
5. [Functions](#functions)
6. [Endpoints](#endpoints)
7. [Configuration](#configuration)
8. [Usage Instructions](#usage-instructions)
9. [Dependencies](#dependencies)
10. [Contributing](#contributing)
11. [License](#license)

## Overview <a name="overview"></a>
Pixel Predictor is a machine learning application designed to predict pixel coordinates in images using deep learning techniques. 

## Architecture <a name="architecture"></a>
The application follows a modular architecture, comprising several components for different stages of the machine learning pipeline:
- **Data Ingestion:** Ingests image data for training and prediction.
- **Model Preparation:** Prepares the base model architecture for training.
- **Model Training:** Trains the model using the prepared data.
- **Prediction:** Predicts pixel coordinates in images using the trained model.
- **Web Interface:** Provides a user-friendly web interface for interacting with the prediction functionality.

## Modules <a name="modules"></a>
- **`pixel_predictor.components`**: Contains the components of the machine learning pipeline.
- **`pixel_predictor.config`**: Defines configuration classes and utilities for managing configurations.
- **`pixel_predictor.pipeline`**: Contains classes for pipeline stages such as prediction.
- **`pixel_predictor.utils`**: Includes utility functions for common tasks such as image preprocessing.

## Classes <a name="classes"></a>
- **`DataIngestion`:** Handles data ingestion tasks, such as loading and preprocessing image data.
- **`PrepareBaseModel`:** Prepares the base model architecture for training by defining the neural network layers.
- **`Training`:** Manages the training process of the model, including compiling the model, training data, and evaluating performance.
- **`PredictionPipeline`:** Performs prediction on images using the trained model.
- **`ConfigurationManager`:** Manages project configurations, including loading and saving configuration files.
- **`ClientApp`:** Represents the web application client for interacting with the prediction functionality.

## Functions <a name="functions"></a>
- **`decodeImage`:** Decodes and saves image data from encoded formats.
- **`preprocess_image`:** Preprocesses input images for prediction including resizing.

## Endpoints <a name="endpoints"></a>
- **`/train`:** Triggers the training process by executing DVC repro command.
- **`/predict`:** Performs prediction on the provided image data.

## Configuration <a name="configuration"></a>
The project uses configuration files (e.g., YAML) to manage various parameters such as file paths, model hyperparameters, etc. Configuration classes are used to load and manage these configurations.

## Usage Instructions <a name="usage-instructions"></a>
1. Clone the repository from GitHub.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Set up the project configurations according to your environment.
4. Run the web application using `python app.py`.
5. Access the web interface and interact with the prediction functionality.

## Dependencies <a name="dependencies"></a>
- Flask: Web framework for building the web application.
- Flask-CORS: Enables Cross-Origin Resource Sharing (CORS) for web requests.
- TensorFlow: Deep learning framework for building and training neural networks.
- NumPy: Library for numerical computations.
- scikit-learn: Library for machine learning algorithms and utilities.
- Pillow: Python Imaging Library for image processing tasks.
- TensorFlow/Keras: High-level neural networks API.



