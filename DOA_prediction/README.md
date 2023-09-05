# DOA Prediction
### 1. 음원 방향 추정
#### 음원의 방향을 네 방향으로 추정한다. 
---
### 2. 개발 환경
Ubuntu 22.04.6  
Python 3.11.3  
PyTorch 2.0.1  
Cuda 12.1

---
### 3. 파일 구조
#### [DOAModel_train.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feat/DOA_prediction/DOA_prediction/DOAModel_train.ipynb)
 ```python
class DirectionDataset(data.Dataset):
 ```
#### 좀 더 강건한 모델을 위해 10~20 사이의 무작위 SNR 비율에 따라 배경 음성을 합성하고 Real-Time Convolutional Neural Network-Based Speech Source Localization on Smartphone에서 제안된 방법에 따라 오디오 데이터를 전처리하여 DataLoader를 구성한다.
---
#### [DOAModel_train.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feat/DOA_prediction/DOA_prediction/DOAModel_train.ipynb)

 ```python
class DOAModel(nn.Module):
 ```
 
#### Real-Time Convolutional Neural Network-Based Speech Source Localization on Smartphone에서 제안된 아키텍쳐를 바탕으로 CNN 기반 음성 방향 인식 모델을 구성한다.
---
#### [DOAModel_train.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feat/DOA_prediction/DOA_prediction/DOAModel_train.ipynb)

 ```python
def train(model, train_loader):
 ```
#### 앞서 구성된 모델을 학습하며 loss가 가장 낮은 모델을 저장한다.
---
#### [doa_prediction_inference.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feat/DOA_prediction/DOA_prediction/doa_prediction_inference.ipynb)

 ```python
def inference(audio):
 ```
#### 음성을 전처리하고 추론한다.
---
### 4. 모델 및 전처리 방법 출처
#### [Real-Time Convolutional Neural Network-Based Speech Source Localization on Smartphone](https://ieeexplore.ieee.org/document/8910614)




