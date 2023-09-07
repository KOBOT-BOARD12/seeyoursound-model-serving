# Sound Classification

### 1. 소리 분류 모델 설명

Denoising된 wav 음성 파일을 받아와 음성을 분류하는 서비스 제공한다.

---

### 2. 개발 환경

Ubuntu 18.04.6  
Python 3.11.3  
PyTorch 2.0.1  
Cuda 11.4

---

### 3. 파일 구조

#### [data_rename.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/main/sound_classification/data_rename.ipynb)

```python
def rename_wav_files(directory_path):
```

모델 학습을 위해 파일명을 재설정해 주는 함수이다. parameter로는 파일의 경로를 설정해 주면 된다.

---

#### [sound_classification.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/main/sound_classification/sound_classification.ipynb)

####

```python
def preprocess_all_wav_files(directory, output_directory, duration=1.0):
def preprocess_and_split_wav(file_path, output_dir, duration=1.0):
```

preprocess_and_split_wav 함수를 통해 모델 학습을 위한 데이터 파일을 생성해 준다. 스트리밍 시에 0.5초씩 1초의 데이터를 만들어 서버에서 wav 파일을 받아오기 때문에 데이터 파일 또한 duration을 1초로 설정하여 split 하였다. preprocess_all_wav_files 함수에서는 파일 경로 내에 있는 모든 파일을 받아와 split 함수를 호출하여 모든 파일에 적용해 준다.

---

#### [sound_classification.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/main/sound_classification/sound_classification.ipynb)

####

```python
class SoundDataset(Dataset):
```

모델 학습을 위한 데이터 셋 클래스이다. 소리 분류를 위한 레이블로 자동차 경적, 개 짖는 소리, 사이렌, 비명 소리, 대화 소리, 침묵 소리를 추가하였다.

---

#### [sound_classification.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/main/sound_classification/sound_classification.ipynb)

####

```python
class SoundClassifier(nn.Module):
```

CNN을 사용한 모델 클래스이다.

---

#### [sound_classification.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/main/sound_classification/sound_classification.ipynb)

####

```python
def train_model(model, dataloader, criterion, optimizer, num_epochs):
```

모델 학습을 진행하는 함수이다. 현재는 overfitting이 발생하지 않도록 epoch을 10으로 설정하여 학습을 진행하였다. 앞으로 프로젝트를 진행하며 데이터가 추가되면 epoch을 변경하며 가장 loss가 적고 정확도가 높은 값을 찾을 예정이다. 함수 내부에서 학습이 완료되면 서빙을 위해 모델을 pt 확장자로 저장해 준다.

---

### 4. 데이터 출처

[자동차 경적, 개 짖는 소리, 사이렌](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=585)  
[비명 소리](https://zenodo.org/record/4844825#.YNv3h-gzZPY)  
[대화 소리](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=568)
