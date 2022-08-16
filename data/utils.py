import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

def spectrogram_padding(spectrogram, spec_len): 
    if spectrogram.shape[1] > spec_len:
        start_cut = (spectrogram.shape[1] - spec_len) // 2 
        return spectrogram[:, start_cut:start_cut+spec_len]

    return np.pad(spectrogram, [[0, 0], [0, spec_len-spectrogram.shape[1]]])

def video_padding(video, video_len): 
    if video.shape[0] > video_len:
        start_cut = (video.shape[0] - video_len) // 2 
        return video[start_cut:start_cut+video_len, :]
    
    return np.pad(video, [[video_len-video.shape[0], 0], [0, 0]])

def get_scaled_bbox(detections, shape):
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
                break
            
            image = image_np.copy()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
            
            pred = face_model.process(image)
            detections = get_scaled_bbox(pred.detections, image.shape)
        
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