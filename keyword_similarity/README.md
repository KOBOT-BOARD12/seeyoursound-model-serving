# Keyword Similarity Prediction
### 1. 키워드 유사도 추정
#### 사람의 말이라 추정된 음성을 받아와 음성의 발음이 키워드와 유사한지 추정한다.
---
### 2. 개발 환경
#### 국민대학교에서 제공하는 Graphic Card가 탑재된 딥러닝 프라이빗 클라우드에서 개발을 진행하였다. Jupyter Lab 환경 위에서 PyTorch와 Python을 사용하였고, ipynb 확장자로 파일을 commit 하였다.
---
### 3. 파일 구조
#### [keyword_similarity.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/keyword_similarity/keyword_similarity/keyword_similarity.ipynb)
 ```python
MFA2IPA: dict
 ```
#### 모델이 추론한 MFA 형식의 발음 기호를 IPA 형식으로 변환하기 위한 딕셔너리이다.
---
#### [keyword_similarity.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/keyword_similarity/keyword_similarity/keyword_similarity.ipynb)

 ```python
def convert_to_ipa(korean_text):
 ```
 
#### 텍스트를 IPA 기호로 변환한다. ex) 영석아 -> jʌŋsʌqɐ
---
#### [keyword_similarity.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/keyword_similarity/keyword_similarity/keyword_similarity.ipynb)

 ```python
def map_to_pred(audio):
 ```
#### 허깅페이스에서 한국어 음성에서 MFA를 추론하도록 학습된 모델로 발음 기호를 추정한다.
---
#### [keyword_similarity.ipynb](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/keyword_similarity/keyword_similarity/keyword_similarity.ipynb)

 ```python
def calculate_similarity(keyword_ipa, predicted_ipa):
 ```
#### Levenshtein 거리로 발음 기호간의 유사도를 계산한다.
---
### 4. 모델 출처
#### [slplab/wav2vec2-xls-r-300m_phone-mfa_korean](https://huggingface.co/slplab/wav2vec2-xls-r-300m_phone-mfa_korean)




