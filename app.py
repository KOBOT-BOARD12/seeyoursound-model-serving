import base64
from fastapi import FastAPI
from pydantic import BaseModel

from utils.utils import bytes_to_wav, denoise_wav
from utils.model_utils import get_audio_classification_class, get_keyword_similarity, get_audio_direction

class PredictionInfo(BaseModel):
    top_channel: str
    bottom_channel: str
    uid: str

app = FastAPI()

@app.post("/prediction")
async def receive_file(audio_data: PredictionInfo):
    data = {"keyword": "unknown"}
    top_channel_data, bottom_channel_data = bytes(base64.b64decode(audio_data.top_channel.encode('utf-8'))), bytes(base64.b64decode(audio_data.bottom_channel.encode('utf-8')))
    uid = audio_data.uid
    bytes_to_wav(uid, top_channel_data, "top_channel.wav")
    bytes_to_wav(uid, bottom_channel_data, "bottom_channel.wav")

    top_channel_audio, bottom_channel_audio_ = denoise_wav(uid)

    class_prediction = get_audio_classification_class(top_channel_audio)

    if class_prediction == None or class_prediction == -1:
        return
    elif class_prediction == 4:
        keyword_prediction = get_keyword_similarity(uid, top_channel_audio)
        data["keyword"] = keyword_prediction

    data["prediction_class"] = str(class_prediction)

    direction = get_audio_direction(bottom_channel_audio_, top_channel_audio)
    data["direction"] = direction

    return data