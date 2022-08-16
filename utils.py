# Модуль, содержащий вспомогательные функции

import os
import subprocess
import cv2
from transliterate import translit
from werkzeug.utils import secure_filename

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

def crop_video(filepath):
    upload_folder, filename = os.path.split(filepath)
    name = filename.split(".")[0]
    temp_filename = name + "_crop.mp4"
    command = f"ffmpeg -ss 0 -to {5} -i {upload_folder}/{filename} -c copy {upload_folder}/{temp_filename}"
    with subprocess.Popen(command, shell=True) as proc:
        if proc.wait() != 0:
            raise Exception("Error when cropping video")
    
    os.remove(f"{upload_folder}/{filename}")
    os.rename(f"{upload_folder}/{temp_filename}", f"{upload_folder}/{filename}")

def convert_file(upload_folder, filename):
    name = filename.split(".")[0]

    if get_video_length_sec(f"{upload_folder}/{filename}") > 5:
        crop_video(f"{upload_folder}/{filename}")

    temp_filename = name + "_conv.mp4"
    command = f"ffmpeg -i {upload_folder}/{filename} -vcodec libx264 -acodec aac {upload_folder}/{temp_filename}"
    with subprocess.Popen(command, shell=True) as proc:
        if proc.wait() != 0:
            raise Exception("Error when converting video")
    
    new_filename = name + ".mp4"
    os.remove(f"{upload_folder}/{filename}")
    os.rename(f"{upload_folder}/{temp_filename}", f"{upload_folder}/{new_filename}")
    return new_filename    