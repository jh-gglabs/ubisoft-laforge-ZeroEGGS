import os
import uvicorn
import json
from json import JSONEncoder
import numpy as np
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.encoders import jsonable_encoder
from inference import ZeroEggsInference


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


app = FastAPI(
    title='ZEROEggs Body Gesture Generation Inference', version='0.0.1-alpha',
    description=''
)


@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Welcome to the ZEROEggs Body Gesture Generation API!'}


@app.post('/predict')
async def get_prediction():
    """
    This endpoint serves the predictions based on the values received from a user and the saved model.

    :param audio file
    :return: 
    """
    # is_valid_file = file.filename.split('.')[-1] in ['wav', 'mp3']
    # if not is_valid_file:
        # return "Check Audio File Format"
    
    network_dir = "../data/outputs/v1/saved_models/"
    config_dir = "../data/processed_v1/"

    zeggs_module = ZeroEggsInference(
        network_directory_path=network_dir,
        config_directory_path=config_dir
    )

    audio_file_path = "../data/samples/067_Speech_2_x_1_0.wav"
    bvh_file_path = "../data/samples/067_Speech_2_x_1_0.bvh"

    prediction = zeggs_module.generate_gesture_from_audio_file(
        audio_file_path, bvh_file_path
    )

    json_str = json.dumps(
        prediction, 
        cls=NumpyArrayEncoder
    ).replace("\"", "")
    
    return json_str


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 6000))
    uvicorn.run(app, host="127.0.0.1", port=port)