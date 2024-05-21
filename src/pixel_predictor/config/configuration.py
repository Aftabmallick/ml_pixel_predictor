import os

from pixel_predictor.constants import *
from pixel_predictor.utils.common import read_yaml, create_directories
from pixel_predictor.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH,
                 ):
            """
        Initialize ConfigurationManager with the file paths for configuration and parameters.

        Args:
        config_filepath (str): Path to the configuration file.
        params_filepath (str): Path to the parameters file.
        """
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
          """
        Get the configuration object for data ingestion.

        Returns:
        DataIngestionConfig: Configuration object for data ingestion.
        """
          config = self.config.data_ingestion

          create_directories([config.root_dir])

          data_ingestion_config = DataIngestionConfig(
                root_dir = config.root_dir,    
                local_data_file = config.local_data_file,
          )
          return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
          """
        Get the configuration object for preparing the base model.

        Returns:
        PrepareBaseModelConfig: Configuration object for preparing the base model.
        """
          config = self.config.prepare_base_model

          create_directories([config.root_dir])

          prepare_base_model_config = PrepareBaseModelConfig(
                root_dir = Path(config.root_dir),
                base_model_path= Path(config.base_model_path),
                params_image_size= self.params.IMAGE_SIZE,
                params_learning_rate= self.params.LEARNING_RATE,
                
          )
          return prepare_base_model_config


    
    def get_training_config(self) -> TrainingConfig:
        """
        Get the configuration object for training.

        Returns:
        TrainingConfig: Configuration object for training.
        """
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = self.config.training.training_data_path
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            base_model_path=Path(prepare_base_model.base_model_path),
            training_data=Path(training_data),
            params_learning_rate= self.params.LEARNING_RATE,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config