# Project 'SeeYourSound' Model Backend

### 1. 'SeeYourSound'에서의 모델 백엔드

모델 서빙과 중앙 제어 서버와의 연결을 맡았다.
<br>

### 2. 'SeeYourSound' 모델 백엔드 개발 환경

Ubuntu 22.04.3, Python 3.11.1 버전에서 개발을 진행하였다.
<br>

### 3. 'SeeYourSound' 모델 백엔드의 구조

a. [Manager](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/blob/develop/manager/firebase_manager.py): Manager Folder - Firebase Firestore와 연동한다.

b. [Utils](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/tree/develop/utils): 오디오 파일 생성, 오디오 디노이징, 모델 추론 등을 진행한다.

c. [Sound Classification](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/tree/develop/sound_classification): Sound Classification에 사용된 학습 데이터 정보, 모델 학습 코드가 포함되어 있다.

d. [TODO](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/tree/develop/sound_classification): TODO

e. [TODO](https://github.com/KOBOT-BOARD12/seeyoursound-model-serving/tree/develop/sound_classification): TODO
<br>

### 4. ENV

```
TYPE
PROJECT_ID
PRIVATE_KEY_ID
PRIVATE_KEY
CLIENT_EMAIL
CLIENT_ID
AUTH_URI
TOKEN_URI
AUTH_PROVIDER_X509_CERT_URL
CLIENT_X509_CERT_URL
UNIVERSE_DOMAIN
CENTRAL_SERVER_URL
```

<br>

### 5-1. How to install (without Docker)

- Firebase 프로젝트 생성 (백앤드와 동일한 Firebase 프로젝트 사용)
  - [공식 문서](https://firebase.google.com/)에 따라 Firebase 프로젝트를 생성한다.
  - ENV 항목을 참고하여 .env 파일을 채워서 firebase 세팅 작업을 진행한다.
- repository clone 받기

```shell
git clone https://github.com/KOBOT-BOARD12/seeyoursound-model-serving.git
```

- 모델 다운 받기

[구글 드라이브](https://drive.google.com/file/d/1KC4cleo_hQfop0Anw3zvXsy25BCl4tI6/view?usp=sharing)로 이동하여 zip 파일을 다운받은 후 프로젝트 루트의 model 파일에 압축 풀기

- Python 가상 환경 설정

```shell
python -m venv .venv
```

```shell
. .venv/bin/activate
```

- 필요한 package 설치

```shell
pip install -r requirements.txt
```

- 실행

```shell
uvicorn app:app --host=0.0.0.0 --port=8001
```

<br>

### 5-2. How to install (with Docker)

- Firebase 프로젝트 생성 (백앤드와 동일한 Firebase 프로젝트 사용)
  - [공식 문서](https://firebase.google.com/)에 따라 Firebase 프로젝트를 생성한다.
  - ENV 항목을 참고하여 .env 파일을 채워서 firebase 세팅 작업을 진행한다.
- repository clone 받기

```shell
git clone https://github.com/KOBOT-BOARD12/seeyoursound-model-serving.git
```

- Docker 세팅 후 실행하기

```shell
docker build mode-backend .
```

```shell
# GPU를 사용하여 모델을 돌릴 경우
docker run -p 8001:8001 --gpus '"device=0"' model-backend

# CPU를 사용하여 모델을 돌릴 경우
docker run -p 8001:8001 model-backend
```
