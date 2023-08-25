import os
import wave
import librosa
import configparser as parser

from Audio_Denoising.denoise import AudioDeNoise

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
    audio_data, _ = librosa.load(file_path)
    return audio_data

def denoise_wav(path):
    audio = AudioDeNoise(path + "/top_channel.wav")
    audio.deNoise(path + "/top_channel.wav")
    audio = AudioDeNoise(path + "/bottom_channel.wav")
    audio.deNoise(path + "/bottom_channel.wav")

    top_channel_audio, _ = librosa.load(path + "/top_channel.wav")
    bottom_channel_audio, _ = librosa.load(path + "/bottom_channel.wav")

    return top_channel_audio, bottom_channel_audio

    