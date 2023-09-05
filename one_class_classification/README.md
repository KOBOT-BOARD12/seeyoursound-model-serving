# One Class Support Vector Machine
### 1. 소리 분류 모델 설명
#### Denoising된 wav 오디오 파일을 받아와 오디오가 학습된 클래스에 포함되는지 OCSVM으로 분류하는 서비스 제공한다. 특징 추출은 AudioSet 데이터셋으로 학습된 distil-ast 모델을 활용하였다.
---
### 2. 개발 환경
#### 국민대학교에서 제공하는 Graphic Card가 탑재된 딥러닝 프라이빗 클라우드에서 개발을 진행하였다. Jupyter Lab 환경 위에서 PyTorch와 Python을 사용하였고, ipynb 확장자로 파일을 commit 하였다.
---
### 3. 파일 구조
#### [train_ocsvm.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feature/ocsvm-train-inference/one_class_classification/train_ocsvm.ipynb)
 ```python
def feature_extraction(data_paths):
 ```
#### 음성 데이터에서 특징을 추출하는 함수이다. 음성 데이터는 비정형 데이터이므로 많은 음성 데이터를 학습한 모델로 특정한 분포에 매핑된 특징을 추출하여 OCSVM을 학습하여야 한다.
---
#### [train_ocsvm.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feature/ocsvm-train-inference/one_class_classification/train_ocsvm.ipynb)

 ```python
def fit_transform_scaler(train_set, test_set):
 ```
 
#### 모델이 학습을 원할히 할 수 있도록 추출된 특징의 스케일을 scikit learn의 RobustScaler로 맞추어준다.
---
#### [train_ocsvm.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feature/ocsvm-train-inference/one_class_classification/train_ocsvm.ipynb)

 ```python
def  fit_transform_pca(train_set, test_set):
 ```
#### 추출된 특징의 차원은 매우 높다. 차원의 저주를 방지하기 위해 PCA로 고차원의 특징을 128차원으로 축소한다. 이 경우 모든 축의 **Explained Variance Ratio**의 합은 0.89이다.
---
#### [train_ocsvm.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feature/ocsvm-train-inference/one_class_classification/train_ocsvm.ipynb)

 ```python
def  param_op(gamma, kernel, nu):
 ```
#### 부르트 포스로 OCSVM의 하이퍼 파라미터인 gamma, kernel, nu의 최적값을 찾는다. 이리하여 구해진 최적의 하이퍼 파라미터는 kernel: rbf , gamma: 0.0001 , nu: 0.156이다. 성능은 accuracy : 0.94, recall : 0.97, precision : 0.92, F1 Score : 0.95이다.
---
#### [inference_ocsvm.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feature/ocsvm-train-inference/one_class_classification/train_ocsvm.ipynb)

 ```python
def  model_load(ast_path, scaler_path, pca_path, ocsvm_path):
 ```
#### 추론시 사용할 모델, 스케일러, PCA 모델, OCSVM을 로드한다.
---
#### [inference_ocsvm.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/feature/ocsvm-train-inference/one_class_classification/train_ocsvm.ipynb)

 ```python
def  inference(audio, sr=16000):
 ```
#### 로드된 모델로 추론을 실행한다.
---
### 4. 데이터 출처
#### [자동차 경적, 개 짖는 소리, 사이렌](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=585)

#### [비명 소리](https://zenodo.org/record/4844825#.YNv3h-gzZPY)

#### [대화 소리](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=568)
