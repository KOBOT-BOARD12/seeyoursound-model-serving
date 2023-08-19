from fastapi import APIRouter, File, UploadFile, HTTPException
import soundfile as sf
import numpy as np
import librosa
import torch
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.defpath(__file__))))
from audio_classification import get_audio_classification_model

classficating_sound_router = APIRouter()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_audio_classification_model()

def read_audio(audio: UploadFile) -> np.ndarray:
    try:
        audio_data, sample_rate = sf.read(audio.file)
    except Exception:
        raise HTTPException(status_code=406, detail="wav 파일을 읽어드리는 데 실패했습니다.")
    return audio_data, sample_rate

@classficating_sound_router.post("/audio_classification", response_model=int)
def audio_type(audio: UploadFile = File(...)):
    try: 
        audio_data, _ = read_audio(audio) 
        sr = 16000
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        target_length = 128
        if mfccs.shape[1] > target_length:
            mfccs = mfccs[:, :target_length]
        else:
            mfccs = np.pad(mfccs, ((0, 0), (0, target_length - mfccs.shape[1])), mode='constant')
        input_tensor = torch.Tensor(mfccs).unsqueeze(0).unsqueeze(1).to(DEVICE)
        inputs = F.interpolate(input_tensor, size=(128, 128)).to(DEVICE)
        with torch.no_grad():
            outputs_probs = model(inputs)
            outputs_probs = F.softmax(outputs_probs, dim=1)
            predicted_class_idx = torch.argmax(outputs_probs, dim=1).item()
        return {"result" : predicted_class_idx}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Audio classification error : " + e)