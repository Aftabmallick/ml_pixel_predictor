import os

from pixel_predictor import logger
from pixel_predictor.entity.config_entity import DataIngestionConfig

import numpy as np
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def create_file(self,num_images=10000)-> str:
        '''
        This method creates a specified number of images with corresponding labels,
        saves them to a file, and compresses the file into a .npz format.

        Args:
        num_images (int): The number of images to generate.

        Returns:
        NULL
        '''
        try:
            images = []
            labels = []
            for _ in range(num_images):
                image = np.zeros((50, 50), dtype=np.float32)
                x, y = np.random.randint(0, 50, size=2)
                image[x, y] = 255
                images.append(image)
                labels.append([x, y])
            data_file_path = self.config.local_data_file
           # os.makedirs(data_file_path,exist_ok=True)
            np.savez_compressed(data_file_path,X=np.array(images), y=np.array(labels))
            logger.info(f"Created image data and stored into file {data_file_path}")

        except Exception as e:
            raise e

