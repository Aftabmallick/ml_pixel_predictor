from pixel_predictor.components.data_ingestion import DataIngestion
from pixel_predictor.config.configuration import ConfigurationManager
from pixel_predictor import logger



STAGE_NAME = "Data ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        """
        Initialize DataIngestionTrainingPipeline class.
        """
        pass
    def main(self):
        """
        Main method to execute the data ingestion stage of the pipeline.
        """
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.create_file()

if __name__ =="__main__":
    try:
        logger.info(f">>>>>>stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e