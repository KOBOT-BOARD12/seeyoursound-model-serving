import json
import httpx
import base64
import requests
from fastapi import FastAPI, Request, BackgroundTasks
from os.path import os, join, dirname
from dotenv import load_dotenv, find_dotenv

from utils.utils import bytes_to_wav, reduce_noise_mfcc_up
from utils.model_utils import get_audio_classification_class, get_keyword_similarity, get_audio_direction, get_oscvm_result

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(find_dotenv())

app = FastAPI()

async def get_model_inference(req):
    top_channel, bottom_channel, uid = req["top_channel"], req["bottom_channel"], req["uid"]
    filtered_class = req["filtered_class"]
    websocket_idx = req["websocket_idx"]

    data = {"keyword": "unknown", "websocket_idx": websocket_idx}

    top_channel_data, bottom_channel_data = bytes(base64.b64decode(top_channel.encode('utf-8'))), bytes(base64.b64decode(bottom_channel.encode('utf-8')))
    top_channel_audio = bytes_to_wav(uid, top_channel_data, "top_channel.wav")
    bottom_channel_audio = bytes_to_wav(uid, bottom_channel_data, "bottom_channel.wav")

    top_channel_audio = reduce_noise_mfcc_up(uid + "/top_channel.wav")
    bottom_channel_audio = reduce_noise_mfcc_up(uid + "/bottom_channel.wav")

    oscvm_prediction = get_oscvm_result(top_channel_audio)

    if oscvm_prediction == -1:
        return

    class_prediction = get_audio_classification_class(top_channel_audio)

    if class_prediction != -1 and (not filtered_class[class_prediction]):
        return

    if class_prediction == -1 or class_prediction == 5:
        return
    elif class_prediction == 4:
        keyword_prediction, flag = get_keyword_similarity(uid, top_channel_audio)
        if flag == 0: 
            return
        data["keyword"] = keyword_prediction

    data["prediction_class"] = str(class_prediction)

    sound_speed = 343  # Speed of sound in m/s
    microphone_distance = 0.16  # Distance between microphones in meters
    max_tau = microphone_distance / sound_speed

    theta = get_audio_direction(bottom_channel_audio, top_channel_audio, fs=16000, max_tau=max_tau)
    theta=int(theta)

    if theta > 0:
        if theta > 45:
            data["direction"] = "북쪽"
        else:
            data["direction"] = "동쪽"
    else:
        if theta < -45:
            data["direction"] = "남쪽"
        else:
            data["direction"] = "서쪽"

    async with httpx.AsyncClient() as client:
        response = await client.post(os.getenv("SERVICE_SERVER_URL") + "/get_model_prediction", json=data)

@app.post("/prediction")
async def return_prediction(audio_data: Request, background_tasks: BackgroundTasks):
    req = await audio_data.json()

    background_tasks.add_task(get_model_inference, req)

    return {"message": "Prediction request received."}
