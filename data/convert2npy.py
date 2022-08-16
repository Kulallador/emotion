import tensorflow as tf
import numpy as np
import os
import tqdm
import librosa
import numpy as np
import argparse
from utils import get_faces_from_video, spectrogram_padding, video_padding

parser = argparse.ArgumentParser()

parser.add_argument("--input", help="path to data")
parser.add_argument("--output", help="path to output data")

def convert_video_data_to_np(data_path, seq_len=157):
    emotion_model = tf.keras.models.load_model("models/feature_extractor.h5")
    feature_extractor = tf.keras.Model(inputs=emotion_model.inputs, outputs=emotion_model.layers[-7].output)

    features = {}

    for filename in tqdm.tqdm(os.listdir(data_path)):
        faces = get_faces_from_video(f"{data_path}/{filename}")
        feature = feature_extractor(np.array(faces))
        pad_feature = video_padding(feature, seq_len)
        video_name = filename.split(".")[0][3:]
        features[video_name] = pad_feature
        
    return features

def convert_audio_data_to_np(data_path, spec_len=100):
    features = {}
    for filename in tqdm.tqdm(os.listdir(data_path)):
            
        amplitudes, sr = librosa.load(f"{data_path}/{filename}", sr=16000, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(y=amplitudes, n_mels=128, sr=sr, fmin=1, fmax=8192)
        pad_mel_spectrogram = spectrogram_padding(mel_spectrogram, spec_len)
        audio_name = filename.split(".")[0][3:]
        features[audio_name] = pad_mel_spectrogram

    return features

if __name__ == "__main__":
    args = parser.parse_args()

    video_data = convert_video_data_to_np(f"{args.input}/video")
    audio_data = convert_audio_data_to_np(f"{args.input}/audio")

    names = list(set(video_data.keys()) & set(audio_data.keys()))
    emotion = {
        "01":"calm", 
        "02":"calm",
        "03":"happy",
        "04":"sad",
        "05":"angry",
        "06":"fearful",
        "07":"disgust",
        "08":"surprised"
    }
    data = np.array([(name, # filename
                    emotion[name.split("-")[1]], # emotion
                    int(name.split("-")[-1]), # actor id
                    video_data[name],  # video features
                    audio_data[name] # audio features
                    ) for name in names])

    with open(f"{args.output}/data.npy", 'wb') as f:
        np.save(f, data)