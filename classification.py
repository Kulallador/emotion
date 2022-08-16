# Модуль, содеращий функции для классификации эмоций

import librosa
import numpy as np
import tensorflow as tf
from data.utils import get_faces_from_video, spectrogram_padding

def emotion_classification(filepath):
    class2idx =  {'Angry':0, 'Calm':1, 'Disgust':2, 'Fearful':3, 'Happy':4, 'Sad':5, 'Surprised':6}
    idx2class = {idx:name for name, idx in class2idx.items()}

    emotion_by_image_model = tf.keras.models.load_model(f"./models/feature_extractor.h5")
    feature_extractor = tf.keras.Model(
        inputs=emotion_by_image_model.inputs, 
        outputs=emotion_by_image_model.layers[-7].output
    )

    classificator = tf.keras.models.load_model(f"./models/rnn_rnn_ravdess_song")

    faces = []
    for mode in [0, 1]:
        faces = get_faces_from_video(filepath, model_selection=mode)
        if len(faces) != 0:
            break

    if len(faces) == 0:
        raise Exception("Face not found")
        
    features = feature_extractor.predict(np.array(faces))
    amplitudes, sr = librosa.load(filepath, sr=16000, mono=True)
    spectrogram = librosa.feature.melspectrogram(y=amplitudes, n_mels=128, sr=sr, fmin=1, fmax=8192)
    spectrogram = spectrogram_padding(spectrogram, 100)

    pred = tf.squeeze(
        classificator.predict(
            (tf.expand_dims(features, axis=0), tf.expand_dims(spectrogram.T, axis=0))
        ),
        axis=0
    )    

    pred_class = np.argmax(pred)
    class_name = idx2class[pred_class]

    return class_name
