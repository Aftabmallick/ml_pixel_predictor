import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pixel_predictor import logging
from PIL import Image

from pixel_predictor.utils.common import preprocess_image



class PredictionPipeline:
    def __init__(self, filename):
        """
        Initialize PredictionPipeline with the provided filename.

        Args:
        filename (str): Path to the image file for prediction.
        """
        self.filename = filename
        

    def predict(self):
        """
        Perform prediction using the loaded model on the provided image.

        Returns:
        str: Predicted coordinates in the format "x, y".
        """
        model = load_model(os.path.join("artifacts/training","model.h5"))
        #image = Image.open(self.filename)
        #image = np.array(image)
        img = preprocess_image(self.filename)
        result = model.predict(img)
        print(result)
        logging.info(f"Succefully predicted{result}")
        x = round(result[0][0],0)
        y = round(result[0][1],0)
        r = str(int(x))+" , "+str(int(y))
        print(r)
        return r

