import cv2
import librosa
from utils import spectrogram_padding
import numpy as np
import tensorflow as tf
import mediapipe as mp

def get_normalize_bbox(detections, shape):
    res = []
    if not detections:
        return res

    for detection in detections:
        relative_bbox = detection.location_data.relative_bounding_box
        bbox = []

        relative_bbox_list = [relative_bbox.xmin, relative_bbox.ymin, relative_bbox.width, relative_bbox.height] 
        for i in range(len(relative_bbox_list)):
            if relative_bbox_list[i] < 0:
                bbox.append(0)
                continue
            
            if i % 2 == 0:
                bbox.append(int(relative_bbox_list[i] * shape[1]))
            else:
                bbox.append(int(relative_bbox_list[i] * shape[0]))
                
        res.append(bbox)
    return res

def get_faces_from_video(filepath, model_selection=0):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=0.5) as face_model:
        cap = cv2.VideoCapture(filepath)

        faces = []
        while True:
            ret, image_np = cap.read()
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            image = image_np.copy()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
            
            pred = face_model.process(image)
            detections = get_normalize_bbox(pred.detections, image.shape)
        
            image.flags.writeable = True

            crop_faces = []
            if detections:
                detections.sort(key= lambda x: x[2]*x[3], reverse=True)

                bbox = detections[0]
                x, y, w, h = bbox
                crop_faces.append(image_np[y:y+h, x:x+w])
            
            if len(crop_faces) > 0 :
                img = tf.image.resize(crop_faces[0], (48,48))
                img = tf.image.rgb_to_grayscale(img)
                img *= 1./255

                faces.append(img)

        cap.release()

        return faces

def emotion_classification(filepath, model_path, classificator_name):
    class2idx =  {'angry':0, 'calm':1, 'disgust':2, 'fearful':3, 'happy':4, 'normal':5, 'sad':6, 'surprised':7}
    idx2class = {idx:name for name, idx in class2idx.items()}


    emotion_by_image_model = tf.keras.models.load_model(f"{model_path}/resnet4.h5")
    feature_extractor = tf.keras.Model(
        inputs=emotion_by_image_model.inputs, 
        outputs=emotion_by_image_model.layers[-7].output
    )

    classificator = tf.keras.models.load_model(f"{model_path}/{classificator_name}")

    faces = []
    for mode in [0, 1]:
        faces = get_faces_from_video(filepath, model_selection=mode)
        if len(faces) != 0:
            break

    if len(faces) == 0:
        raise Exception("Лицо не найдено")
        

    features = feature_extractor.predict(np.array(faces))
    amplitudes, sr = librosa.load(filepath, sr=16000, mono=True)
    spectrogram = librosa.feature.melspectrogram(y=amplitudes, n_mels=128, sr=sr, fmin=1, fmax=8192)
    spectrogram = spectrogram_padding(spectrogram, 100)

    pred = tf.squeeze(
        classificator.predict(
            (tf.expand_dims(features, axis=0), tf.expand_dims(spectrogram, axis=0))
        ),
        axis=0
    )

    pred_class = np.argmax(pred)
    class_name = idx2class[pred_class]
    return class_name
