import math
import torch
import librosa
import Levenshtein
import numpy as np
import configparser as parser
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from manager.firebase_manager import db
from model.audio_classification import SoundClassifier

MFA2IPA = {
    'A': 'ɐ',
    'iA': 'jɐ',
    'oA': 'wɐ',
    'E': 'ɛ',
    'iE': 'je',
    'oE': 'wɛ',
    'I': 'i',
    'uI': 'wi',
    'O': 'o',
    'iO': 'jo',
    'U': 'u',
    'iU': 'ju',
    'EO': 'ʌ',
    'iEO': 'jʌ',
    'uEO': 'wʌ',
    'EU': 'ɯ',
    'euI': 'ɯj',
    'G': 'q',
    'N': 'n',
    'D': 'd',
    'R': 'ɾ',
    'M': 'm',
    'B': 'p',
    'S': 's',
    'J': 'tɕ',
    'Kh': 'kh',
    'Th': 'th',
    'Ph': 'ph',
    'H': 'h',
    'GG': 'qo',
    'DD': 't',
    'CHh': 'tʃh',
    'NG': 'ŋ',
    'L':'ɫ',
    'p':'p',
    'k': 'k',
    'JJ':'tɕ',
    'SS':'s',
    'BB':'p',
}

properties = parser.ConfigParser()
properties.read("config.ini")
wav_detail = properties["WAV_DETAIL"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audio_classification_model = SoundClassifier(num_classes=5)
audio_classification_model.load_state_dict(torch.load("./model/audio_classification.pt", map_location=DEVICE))
audio_classification_model.to(DEVICE)
audio_classification_model.eval()

keyword_processor = Wav2Vec2Processor.from_pretrained("slplab/wav2vec2-xls-r-300m_phone-mfa_korean")
keyword_model = Wav2Vec2ForCTC.from_pretrained("slplab/wav2vec2-xls-r-300m_phone-mfa_korean").to(DEVICE)

def get_audio_classification_class(audio_file):
    try: 
        mfccs = librosa.feature.mfcc(y=audio_file, sr=int(wav_detail["sample_rate"]), n_mfcc=13)
        target_length = 128
        if mfccs.shape[1] > target_length:
            mfccs = mfccs[:, :target_length]
        else:
            mfccs = np.pad(mfccs, ((0, 0), (0, target_length - mfccs.shape[1])), mode='constant')
        input_tensor = torch.Tensor(mfccs).unsqueeze(0).unsqueeze(0).to(DEVICE)
        inputs = F.interpolate(input_tensor, size=(128, 128)).to(DEVICE)
        with torch.no_grad():
            outputs_probs = audio_classification_model(inputs)
            outputs_probs = F.softmax(outputs_probs, dim=1)
            predicted_class_idx = torch.argmax(outputs_probs, dim=1).item()
            return predicted_class_idx
    except Exception as e:
        print('Error occur in audio classifiaction : ', e)
        return -1

def map_to_pred(audio):
    inputs = keyword_processor(audio, sampling_rate=int(wav_detail["sample_rate"]), return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        logits = keyword_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = keyword_processor.batch_decode(predicted_ids)
    return transcription 

def calculate_similarity(keyword_ipa, predicted_ipa):
    distance = 0
    keyword_length = len(keyword_ipa)
    prediction_length = len(predicted_ipa)

    if keyword_length >= prediction_length:
        padded_keyword_ipa = keyword_ipa+'0' * (15 - keyword_length) if keyword_length < 15 else keyword_ipa
        padded_predicted_ipa = predicted_ipa + '0' * (15 - prediction_length) if prediction_length < 15 else predicted_ipa
        distance = Levenshtein.ratio(padded_keyword_ipa, padded_predicted_ipa)
    else:
        for window in range(len(predicted_ipa) - keyword_length):
            padded_keyword_ipa = keyword_ipa + '0' * (15 - keyword_length) if keyword_length < 15 else keyword_ipa
            padded_predicted_ipa = predicted_ipa[window : window + keyword_length + 1] + '0' * (15 - keyword_length + 1) if keyword_length + 1 < 15 else predicted_ipa[window : window + keyword_length + 1]
            distance = max(distance, Levenshtein.ratio(padded_keyword_ipa, padded_predicted_ipa))
    
    return distance

def get_keyword_similarity(uid, audio_file):
    uid_ref = db.collection("Users").document(uid)
    doc = uid_ref.get()
    if doc.exists:
        max_similarity = 0
        max_similarity_keyword = ""
        existing_keywords = doc.to_dict().get("keywords", [])
        ipa_keywords = list(existing_keywords.values())
        keywords = list(existing_keywords.keys())
        result = map_to_pred([audio_file])
        model_output_ipa = ''.join([MFA2IPA[mfa] for mfa in result[0].split(' ')])
        for i in range(len(ipa_keywords)):
            ipa_of_keyword = ipa_keywords[i]
            similarity = calculate_similarity(ipa_of_keyword, model_output_ipa)
            if max_similarity < similarity:
                max_similarity = similarity
                max_similarity_keyword = keywords[i]
        return max_similarity_keyword

def get_audio_direction(sig, refsig, fs=1, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)  
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:    
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)

    theta = math.asin(tau / max_tau) * 180 / math.pi
    return theta
 