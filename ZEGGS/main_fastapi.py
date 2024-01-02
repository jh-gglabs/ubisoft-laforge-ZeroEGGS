from io import BytesIO
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

bvh_file_mappers = {
    "0": os.path.join("../data/samples", "030_Agreement_0_x_1_0.bvh"),
    "1": os.path.join("../data/samples", "026_Angry_0_x_1_0.bvh"),
    "2": os.path.join("../data/samples", "031_Disagreement_0_x_1_0.bvh"),
    "3": os.path.join("../data/samples", "045_Distracted_0_x_1_0.bvh"),
    "4": os.path.join("../data/samples", "037_Flirty_1_x_1_0.bvh"),
    "5": os.path.join("../data/samples", "011_Happy_0_x_1_0.bvh"),
    "6": os.path.join("../data/samples", "057_Laughing_0_x_1_0.bvh"),
    "7": os.path.join("../data/samples", "065_Speech_0_x_1_0.bvh"),
    "8": os.path.join("../data/samples", "001_Neutral_0_x_1_0.bvh"),
    "9": os.path.join("../data/samples", "021_Old_0_x_1_0.bvh"),
    "10": os.path.join("../data/samples", "039_Pensive_0_x_1_0.bvh"),
    "11": os.path.join("../data/samples", "016_Relaxed_0_x_1_0.bvh"),
    "12": os.path.join("../data/samples", "006_Sad_0_x_1_0.bvh"),
    "13": os.path.join("../data/samples", "048_Sarcastic_0_x_1_0.bvh"),
    "14": os.path.join("../data/samples", "042_Scared_0_x_1_0.bvh"),
    "15": os.path.join("../data/samples", "059_Sneaky_0_x_1_0.bvh"),
    "16": os.path.join("../data/samples", "054_Still_0_x_1_0.bvh"),
    "17": os.path.join("../data/samples", "051_Threatening_0_x_1_0.bvh"),
    "18": os.path.join("../data/samples", "063_Tired_1_x_1_0.bvh"),
}


app = FastAPI(
    title='ZEROEggs Body Gesture Generation Inference', version='0.0.1-alpha',
    description=''
)

@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Welcome to the ZEROEggs Body Gesture Generation API!'}


@app.post('/predict')
async def get_prediction(file: UploadFile = File(...), style_id: int = 8):
    """
    This endpoint serves the predictions based on the values received from a user and the saved model.

    :param 
        audio_file
        style_id: 0 ~ 18
            0: agreement | 1: angry | 2: disagreement | 3: distracted | 4: flirty |
            5: happy | 6: laughing | 7: oration | 8: neutral | 9: old |
            10: pensive | 11: relaxed | 12: sad | 13: sarcastic | 14: scared |
            15: sneaky | 16: still | 17: threatening | 18: tired
    :return: bvh_strings
    """

    network_dir = "../data/outputs/v1/saved_models/"
    config_dir = "../data/processed_v1/"

    zeggs_module = ZeroEggsInference(
        network_directory_path=network_dir,
        config_directory_path=config_dir
    )

    style_id = str(style_id)
    bvh_file_path = bvh_file_mappers.get(style_id)

    bvh_strings = zeggs_module.generate_gesture_from_audio_file(
        await file.read(), bvh_file_path
    )
    
    return bvh_strings


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 6000))
    uvicorn.run(app, host="127.0.0.1", port=port)