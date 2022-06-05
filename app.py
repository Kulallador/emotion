import json
import os
import json
from classification import emotion_classification
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, url_for

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config')
app.config.from_pyfile('config.py')
app.add_url_rule("/files/<name>", endpoint="download_file", build_only=True)
port = int(os.environ.get("PORT", 5000))

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.mkdir(app.config["UPLOAD_FOLDER"])

def validate_file(filename):
    val_types = ["mp4", "flv"]
    file_type = filename.split(".")[-1]

    return file_type in val_types

def save_file(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    return filename 

def convert_file(filename):
    upload_folder = app.config["UPLOAD_FOLDER"]
    name = filename.split(".")[0]
    temp_filename = name + "_conv.mp4"
    err_code = os.system(f"ffmpeg -i {upload_folder}/{filename} -vcodec libx264 -acodec aac {upload_folder}/{temp_filename}")
    if err_code != 0:
        raise Exception("Ошибка при конвертации видео")
    
    new_filename = name + ".mp4"
    os.remove(f"{upload_folder}/{filename}")
    os.rename(f"{upload_folder}/{temp_filename}", f"{upload_folder}/{new_filename}")
    return new_filename

@app.route("/")
def get_index():
    return render_template("index.html")

@app.route("/emotion", methods=["POST"])
def upload_file():
    if "myFile" not in request.files:
        print("file not found")
        return json.dumps({
            "url":None, 
            "emotion":None, 
            "error": "Файл не обнаружен в запросе"
        })
    
    file = request.files["myFile"]
    if file.filename == "":
        print("bad filename")
        return json.dumps({
            "url":None, 
            "emotion":None, 
            "error": f"Файл не выбран"
        })

    if not validate_file(file.filename):
        print("File type not supported")
        return json.dumps({
            "url":None, 
            "emotion":None, 
            "error": f"Тип файла не поддерживается"
        })
    
    filename = save_file(file)
    file.close()

    emotion = ""
    try:
        filename = convert_file(filename)
        upload_folder = app.config["UPLOAD_FOLDER"]
        emotion = emotion_classification(
            f"{upload_folder}/{filename}",
            app.config["MODEL_PATH"],
            app.config["CLASSIFICATOR"]
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