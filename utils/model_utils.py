import math
import torch
import librosa
import Levenshtein
import numpy as np
import joblib
import configparser as parser
import torch.nn.functional as F
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoFeatureExtractor,
    ASTModel,
)
from manager.firebase_manager import db
from model.audio_classification import SoundClassifier
from model.audio_direction import SoundDirection

MFA2IPA = {
    "A": "ɐ",
    "iA": "jɐ",
    "oA": "wɐ",
    "E": "ɛ",
    "iE": "je",
    "oE": "wɛ",
    "I": "i",
    "uI": "wi",
    "O": "o",
    "iO": "jo",
    "U": "u",
    "iU": "ju",
    "EO": "ʌ",
    "iEO": "jʌ",
    "uEO": "wʌ",
    "EU": "ɯ",
    "euI": "ɯj",
    "G": "q",
    "N": "n",
    "D": "d",
    "R": "ɾ",
    "M": "m",
    "B": "p",
    "S": "s",
    "J": "tɕ",
    "Kh": "kh",
    "Th": "th",
    "Ph": "ph",
    "H": "h",
    "GG": "qo",
    "DD": "t",
    "CHh": "tʃh",
    "NG": "ŋ",
    "L": "ɫ",
    "p": "p",
    "k": "k",
    "JJ": "tɕ",
    "SS": "s",
    "BB": "p",
}

properties = parser.ConfigParser()
properties.read("config.ini")
wav_detail = properties["WAV_DETAIL"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_classification_model = SoundClassifier(num_classes=6)
audio_classification_model.load_state_dict(
    torch.load("./model/audio_classification.pt", map_location=DEVICE)
)
audio_classification_model.to(DEVICE)
audio_classification_model.eval()

keyword_processor = Wav2Vec2Processor.from_pretrained(
    "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"
)
keyword_model = Wav2Vec2ForCTC.from_pretrained(
    "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"
).to(DEVICE)

feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")
model = ASTModel.from_pretrained("bookbot/distil-ast-audioset").to(DEVICE)

scaler = joblib.load("./model/robustscaler_AST.pkl")
pca = joblib.load("./model/pca_AST.pkl")
ocsvm = joblib.load("./model/ocsvm_AST.pkl")

audio_direction_model = SoundDirection()
audio_direction_model.load_state_dict(
    torch.load("./model/audio_direction.pt", map_location=DEVICE)
)
audio_direction_model.to(DEVICE)
audio_direction_model.eval()


def get_audio_classification_class(audio_file):
    try:
        mfccs = librosa.feature.mfcc(
            y=audio_file, sr=int(wav_detail["sample_rate"]), n_mfcc=13
        )
        target_length = 128
        if mfccs.shape[1] > target_length:
            mfccs = mfccs[:, :target_length]
        else:
            mfccs = np.pad(
                mfccs, ((0, 0), (0, target_length - mfccs.shape[1])), mode="constant"
            )
        input_tensor = torch.Tensor(mfccs).unsqueeze(0).unsqueeze(0).to(DEVICE)
        inputs = F.interpolate(input_tensor, size=(128, 128)).to(DEVICE)
        with torch.inference_mode():
            outputs_probs = audio_classification_model(inputs)
            outputs_probs = F.softmax(outputs_probs, dim=1)
            predicted_class_idx = torch.argmax(outputs_probs, dim=1).item()
            return predicted_class_idx
    except Exception as e:
        print("Error occur in audio classifiaction : ", e)
        return -1


def map_to_pred(audio):
    inputs = keyword_processor(
        audio,
        sampling_rate=int(wav_detail["sample_rate"]),
        return_tensors="pt",
        padding="longest",
    )
    input_values = inputs.input_values.to(DEVICE)

    with torch.inference_mode():
        logits = keyword_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = keyword_processor.batch_decode(predicted_ids)
    return transcription


def calculate_similarity(keyword_ipa, predicted_ipa):
    distance = 0
    padding_length = 15
    keyword_length = len(keyword_ipa)
    prediction_length = len(predicted_ipa)

    if keyword_length >= prediction_length:
        padded_keyword_ipa = (
            keyword_ipa + "0" * (padding_length - keyword_length)
            if keyword_length < padding_length
            else keyword_ipa
        )
        padded_predicted_ipa = (
            predicted_ipa + "0" * (padding_length - prediction_length)
            if prediction_length < padding_length
            else predicted_ipa
        )
        distance = Levenshtein.ratio(padded_keyword_ipa, padded_predicted_ipa)
    else:
        for window in range(len(predicted_ipa) - keyword_length):
            padded_keyword_ipa = (
                keyword_ipa + "0" * (padding_length - keyword_length)
                if keyword_length < padding_length
                else keyword_ipa
            )
            padded_predicted_ipa = (
                predicted_ipa[window : window + keyword_length + 1]
                + "0" * (padding_length - keyword_length + 1)
                if keyword_length + 1 < padding_length
                else predicted_ipa[window : window + keyword_length + 1]
            )
            distance = max(
                distance, Levenshtein.ratio(padded_keyword_ipa, padded_predicted_ipa)
            )

    return distance


def get_keyword_similarity(uid, audio_file):
    flag = 0
    uid_ref = db.collection("Users").document(uid)
    doc = uid_ref.get()
    if doc.exists:
        max_similarity = 0
        max_similarity_keyword = ""
        existing_keywords = doc.to_dict().get("keywords", [])
        if not bool(existing_keywords):
            return max_similarity_keyword, flag
        ipa_keywords = list(existing_keywords.values())
        keywords = list(existing_keywords.keys())
        result = map_to_pred([audio_file])
        try:
            model_output_ipa = "".join([MFA2IPA[mfa] for mfa in result[0].split(" ")])
            for i in range(len(ipa_keywords)):
                ipa_of_keyword = ipa_keywords[i]
                similarity = calculate_similarity(ipa_of_keyword, model_output_ipa)
                if max_similarity < similarity:
                    max_similarity = similarity
                    max_similarity_keyword = keywords[i]
        except KeyError as e:
            print("Error occur in MFA2IPA : ", e)
        if max_similarity >= 0.8:
            flag = 1

        return max_similarity_keyword, flag


def get_oscvm_result(y):
    input_tensor = feature_extractor(
        y, sampling_rate=int(wav_detail["sample_rate"]), return_tensors="pt"
    )
    with torch.inference_mode():
        feature = model(**input_tensor.to(DEVICE)).last_hidden_state.detach()
        feature = feature.cpu().numpy().reshape(1, -1)
        feature = scaler.transform(feature)
        feature = pca.transform(feature)
        pred = ocsvm.predict(feature)
    return pred


def get_audio_direction(top_channel_audio, bottom_channel_audio):
    with torch.inference_mode():
        stft1 = librosa.stft(
            top_channel_audio, n_fft=512, hop_length=320, win_length=320
        )
        stft2 = librosa.stft(
            bottom_channel_audio, n_fft=512, hop_length=320, win_length=320
        )

        feature_set1 = torch.tensor(
            np.concatenate((stft1.real, stft2.real), axis=0)
        ).to(DEVICE)
        feature_set2 = torch.tensor(
            np.concatenate((stft1.imag, stft2.imag), axis=0)
        ).to(DEVICE)

        feature = torch.concatenate((feature_set1, feature_set2), axis=0).to(DEVICE)

        feature = feature.unsqueeze(0).unsqueeze(0).to(DEVICE)
        direction = audio_direction_model(feature).argmax(dim=1).item()
    if direction == 0:
        return "서쪽"
    elif direction == 1:
        return "남쪽"
    elif direction == 2:
        return "동쪽"
    elif direction == 3:
        return "북쪽"
