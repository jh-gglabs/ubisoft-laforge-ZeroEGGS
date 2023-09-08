import os
import uvicorn
import json
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.encoders import jsonable_encoder
from inference import ZeroEggsInference


app = FastAPI(
    title='ZEROEggs Body Gesture Generation Inference', version='0.0.1-alpha',
    description=''
)


@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Welcome to the ZEROEggs Body Gesture Generation API!'}


@app.post('/predict')
async def get_prediction(file: UploadFile = File(...)):
    """
    This endpoint serves the predictions based on the values received from a user and the saved model.

    :param audio file
    :return: 
    """
    is_valid_file = file.filename.split('.')[-1] in ['wav', 'mp3']
    if not is_valid_file:
        return "Check Audio File Format"
    
    network_dir = "../data/outputs/v1/saved_models/"
    config_dir = "../data/processed_v1/"

    zeggs_module = ZeroEggsInference(
        network_directory_path=network_dir,
        config_directory_path=config_dir
    )

    prediction = zeggs_module.generate_gesture_from_audio_file(await file.read())
    json_str = json.dumps(prediction).replace("\"", "")
    
    return json_str


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)