from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS,cross_origin

from src.pixel_predictor.pipeline.prediction import PredictionPipeline
from src.pixel_predictor.utils.common import decodeImage



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


app =Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self) :
        """
        Initialize ClientApp with a default filename and PredictionPipeline instance.
        """
        self.filename = "inputImage.jpg"
        self.regressor = PredictionPipeline(self.filename)

@app.route("/",methods = ['GET'])
@cross_origin()
def home():
    """
    Home route to render the index.html template.
    """
    return render_template('index.html')


@app.route("/train",methods = ['GET','POST'])
@cross_origin()
def trainRoute():
    """
    Route to trigger the training process by executing the DVC repro command.
    """
    os.system("dvc repro")
    return "Training done successfully"

@app.route("/predict",methods = ['POST'])
@cross_origin()
def predictRoute():
    """
    Route to perform prediction on the provided image data.
    """
    image = request.json['image']
    decodeImage(image,clApp.filename)  
    result = clApp.regressor.predict()
    return jsonify(result)


if __name__ =="__main__":
    # Initialize the ClientApp instance
    clApp = ClientApp()
    app.run(host='0.0.0.0',port = 8080)



