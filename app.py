import base64
from fastapi import FastAPI, Request
from os.path import os, join, dirname
from dotenv import load_dotenv, find_dotenv

from utils.utils import bytes_to_wav
from utils.model_utils import get_audio_classification_class, get_keyword_similarity, get_audio_direction
from utils.model_utils import read_file, reduce_noise_mfcc_up, trim_silence, output_file

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(find_dotenv())

app = FastAPI()

@app.post("/prediction")
async def receive_file(audio_data: Request):
    req = await audio_data.json()
    top_channel, bottom_channel, uid = req["top_channel"], req["bottom_channel"], req["uid"]
    data = {"keyword": "unknown"}

    top_channel_data, bottom_channel_data = bytes(base64.b64decode(top_channel.encode('utf-8'))), bytes(base64.b64decode(bottom_channel.encode('utf-8')))
    top_channel_audio = bytes_to_wav(uid, top_channel_data, "top_channel.wav")
    bottom_channel_audio = bytes_to_wav(uid, bottom_channel_data, "bottom_channel.wav")

    # top_channel_audio, bottom_channel_audio_ = denoise_wav(uid)
    file = [top_channel_audio, bottom_channel_audio]
    for filename in file:
        y, sr = read_file(file)
        y_reduced_mfcc_up = reduce_noise_mfcc_up(y, sr)
        y_reduced_mfcc_up, time_trimmed = trim_silence(y_reduced_mfcc_up)
        output_file('./src/', file, y_reduced_mfcc_up, sr, '_mfcc_up')


    class_prediction = get_audio_classification_class(top_channel_audio)

    if class_prediction == None or class_prediction == -1:
        return
    elif class_prediction == 4:
        keyword_prediction = get_keyword_similarity(uid, top_channel_audio)
        data["keyword"] = keyword_prediction

    data["prediction_class"] = str(class_prediction)

    direction = get_audio_direction(bottom_channel_audio, top_channel_audio)
    data["direction"] = direction

    requests.post(os.getenv("SERVICE_SERVER_URL") + "/get_model_prediction", data=json.dumps(data), headers={"Content-Type": "application/json"})

    return