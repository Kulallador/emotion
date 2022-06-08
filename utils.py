# Модуль, содержащий вспомогательные функции

import os
import subprocess
import numpy as np
import cv2
from transliterate import translit
from werkzeug.utils import secure_filename

def spectrogram_padding(spectrogram, spec_len): 
    if spectrogram.shape[1] > spec_len:
        start_cut = (spectrogram.shape[1] - spec_len) // 2 
        return spectrogram[:, start_cut:start_cut+spec_len]

    return np.pad(spectrogram, [[0, 0], [0, spec_len-spectrogram.shape[1]]])

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

def validate_file(filename):
    val_types = ["mp4", "flv", "mkv", "avi", "wmv"]
    file_type = filename.split(".")[-1]

    return file_type in val_types

def save_file(upload_folder, file):
    filename = translit(file.filename, "ru", reversed=True)
    filename = secure_filename(filename)
    file.save(os.path.join(upload_folder, filename))
    return filename 

def get_video_length_sec(filepath):
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return frame_count / fps

def cut_video(filepath):
    upload_folder, filename = os.path.split(filepath)
    name = filename.split(".")[0]
    temp_filename = name + "_cat.mp4"
    command = f"ffmpeg -ss 0 -to {5} -i {upload_folder}/{filename} -c copy {upload_folder}/{temp_filename}"
    with subprocess.Popen(command, shell=True) as proc:
        if proc.wait() != 0:
            raise Exception("Ошибка при обрезании видео")
    
    os.remove(f"{upload_folder}/{filename}")
    os.rename(f"{upload_folder}/{temp_filename}", f"{upload_folder}/{filename}")


def convert_file(upload_folder, filename):
    name = filename.split(".")[0]

    if get_video_length_sec(f"{upload_folder}/{filename}") > 5:
        cut_video(f"{upload_folder}/{filename}")

    temp_filename = name + "_conv.mp4"
    command = f"ffmpeg -i {upload_folder}/{filename} -vcodec libx264 -acodec aac {upload_folder}/{temp_filename}"
    with subprocess.Popen(command, shell=True) as proc:
        if proc.wait() != 0:
            raise Exception("Ошибка при конвертации видео")
    
    new_filename = name + ".mp4"
    os.remove(f"{upload_folder}/{filename}")
    os.rename(f"{upload_folder}/{temp_filename}", f"{upload_folder}/{new_filename}")
    return new_filename    