from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pixel_predictor.entity.config_entity import TrainingConfig
from tensorflow.keras.optimizers import Adam


class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initialize Training class with the provided configuration.

        Args:
        config (TrainingConfig): Configuration object containing training parameters.
        """
        self.config = config

    
    def get_base_model(self):
        """
        Load the pre-trained base model from the specified path.
        """
        self.model = tf.keras.models.load_model(
            self.config.base_model_path
        )
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model to the specified path.

        Args:
        path (Path): Path where the model should be saved.
        model (tf.keras.Model): Trained model to be saved.
        """
        model.save(path)
    def train(self):
        """
        Train the model using the specified configuration.
        """
        data = np.load(self.config.training_data)
        X, y = data['X'], data['y']
        X = X.reshape((X.shape[0], 50, 50, 1))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.compile(optimizer=Adam(learning_rate = self.config.params_learning_rate),loss='mean_squared_error')
        self.model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=self.config.params_epochs,
            batch_size=self.config.params_batch_size
        )
        self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )