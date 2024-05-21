from pixel_predictor.components.prepare_base_model import PrepareBaseModel
from pixel_predictor.config.configuration import ConfigurationManager
from pixel_predictor import logger
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
STAGE_NAME= "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self) :
        """
        Initialize PrepareBaseModelTrainingPipeline class.
        """
        pass
    def main(self):
        """
        Main method to execute the preparation of the base model stage of the pipeline.
        """
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config = prepare_base_model_config)
        prepare_base_model.get_base_model()
if __name__ =='__main__':
    try:
        logger.info(f"****************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise e
