# Модуль веб-сервера. Содержит роуты для обработки запросов.

import os
import subprocess
import json
from utils import validate_file, save_file, convert_file
from classification import emotion_classification
from flask import Flask, render_template, request, send_from_directory, url_for

app = Flask(__name__)
app.config.from_object('config')
app.config.from_pyfile('config.py')
app.add_url_rule("/files/<name>", endpoint="download_file", build_only=True)

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.mkdir(app.config["UPLOAD_FOLDER"])
    
port = int(os.environ.get("PORT", 5000))

subprocess.Popen(["ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

@app.route("/")
def get_index():
    return render_template("index.html")

@app.route("/emotion", methods=["POST"])
def process_file():
    if "myFile" not in request.files:
        print("file not found")
        return json.dumps({
            "url":None, 
            "emotion":None, 
            "error": "File not found in request"
        })
    
    file = request.files["myFile"]
    if file.filename == "":
        print("bad filename")
        return json.dumps({
            "url":None, 
            "emotion":None, 
            "error": f"File not selected"
        })

    if not validate_file(file.filename):
        print("File type not supported")
        return json.dumps({
            "url":None, 
            "emotion":None, 
            "error": f"File type not supported"
        })

    upload_folder = app.config["UPLOAD_FOLDER"]
    
    filename = save_file(upload_folder, file)
    file.close()

    emotion = ""
    try:
        filename = convert_file(upload_folder, filename)
        emotion = emotion_classification(
            f"{upload_folder}/{filename}"
        )

    except Exception as e:
        print(e)
        return json.dumps({
            "url": None, 
            "emotion":emotion, 
            "error":str(e)
        })
    
    url = url_for("download_file", name=filename)
    return json.dumps({
        "url": url, 
        "emotion":emotion, 
        "error":None
    })

@app.route("/files/<name>")
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=port)