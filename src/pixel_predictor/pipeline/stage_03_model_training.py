from pixel_predictor.config.configuration import ConfigurationManager
from pixel_predictor import logger
import os

from pixel_predictor.components.model_training import Training 

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        """
        Initialize ModelTrainingPipeline class.
        """
        pass
    def main(self):
        """
        Main method to execute the model training stage of the pipeline.
        """
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train()

if __name__ =='__main__':
    try:
        logger.info(f"****************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise e