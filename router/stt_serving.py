from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
from transformers import pipeline
import numpy as np
import librosa

stt_serving_router = APIRouter()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="CookieMonster99/whisper-small-KR",
  chunk_length_s=30,
  device=device,
)

def read_sound(sound: UploadFile) -> np.ndarray:
  try:
    sound_data, sr = librosa.load(sound.file, sr=16000)
  except Exception as e:
    raise HTTPException(status_code=400, detail="파일")
  return sound_data

@stt_serving_router.post("/convert_sound", response_model=str)
def convert_sound(sound: UploadFile = File(...)):
  sound_data = read_sound(sound)
  prediction = pipe(sound_data, batch_size=1)
  return prediction