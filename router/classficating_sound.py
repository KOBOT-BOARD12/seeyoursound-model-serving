from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import soundfile as sf
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import soundfile
import torch.nn.functional as F

classficating_sound_router = APIRouter()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def read_audio(audio: UploadFile) -> np.ndarray:
    try:
        audio_data, sample_rate = sf.read(audio.file)
    except Exception as e:
        raise HTTPException(status_code=406, detail="wav 파일을 읽어드리는 데 실패했습니다.")
    return audio_data, sample_rate

class SoundClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SoundClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # 중간의 FC 계층 추가
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)   # 마지막 FC 계층
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SoundClassifier(num_classes=5)
model.load_state_dict(torch.load("0818model.pt", map_location="cpu"))
model.eval()

@classficating_sound_router.post("/audio_type", response_model=int)
def audio_type(audio: UploadFile = File(...)):
    audio_data, _ = read_audio(audio) 
    sr = 16000  # 샘플링 레이트 설정 (필요에 따라 수정)
    target_duration = 2.0  # 2초로 설정

    audio_array = audio_data

    target_length = int(target_duration * sr)
    padded_audio = np.pad(audio_array, (0, max(0, target_length - len(audio_array))), mode='constant')
    input_tensor = torch.Tensor(padded_audio).to("cpu")
    input_tensor = input_tensor.view(1, 1, -1)
    inputs = F.interpolate(input_tensor.unsqueeze(1), size=(128, 128)).to("cpu")

    with torch.no_grad():
        outputs_probs = model(inputs)
        outputs_probs = F.softmax(outputs_probs, dim=1)
        predicted_class_idx = torch.argmax(outputs_probs, dim=1).item()
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Class probabilities: {outputs_probs}")
        return predicted_class_idx