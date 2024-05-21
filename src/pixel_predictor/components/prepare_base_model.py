import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from pixel_predictor.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize PrepareBaseModel class with the provided configuration.

        Args:
        config (PrepareBaseModelConfig): Configuration object containing model parameters.
        """
        self.config = config
    def get_base_model(self):
        """
        Define and save the base model with the specified architecture.

        The model architecture consists of:
        - Two convolutional layers with ReLU activation
        - Flatten layer to convert the 2D output to 1D
        - Two dense layers with ReLU activation, followed by a final dense layer without activation
          (output layer for prediction).

        The model is saved to the specified path.
        """
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.config.params_image_size),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2)
        ])

        self.save_model(path=self.config.base_model_path, model=self.model)
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the model to the specified path.

        Args:
        path (Path): Path where the model should be saved.
        model (tf.keras.Model): Model to be saved.
        """
        model.save(path)

    