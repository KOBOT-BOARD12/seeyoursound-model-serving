import os
import wave
import librosa
import python_speech_features
import configparser as parser
from pysndfx import AudioEffectsChain

properties = parser.ConfigParser()
properties.read("config.ini")
wav_detail = properties["WAV_DETAIL"]

def bytes_to_wav(path, bytes_data, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path + "/" + file_name
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(int(wav_detail["num_channels"]))
        wf.setsampwidth(int(wav_detail["sample_width"]))
        wf.setframerate(int(wav_detail["sample_rate"]))
        wf.writeframes(bytes_data)
    wf.close()

def reduce_noise_mfcc_up(path):
    y, sr = librosa.load(path)

    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(frequency=min_hz*(-1), gain=12.0, slope=0.5)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)