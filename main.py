import shutil
import urllib.request

from fastapi import FastAPI, UploadFile, File
from checking_yolo5 import DetectObject

app = FastAPI()


@app.post("/upload_file/")
async def create_upload_file(file: UploadFile = File(...)):
    detect = DetectObject()
    path = file.filename
    with open(f'{path}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    images = detect.video_to_images(file.filename)
    result = detect.detect_objects(images)
    return result


@app.post("/link_upload_file/")
def upload_file():
    detect = DetectObject()
    file_name = 'trial_video.mp4'
    dwn_link = "https://www.youtube.com/watch?v=Glk4det3JFE&ab_channel=AlphaCreations"
    urllib.request.urlretrieve(dwn_link, file_name)

    images = detect.video_to_images(file_name)
    result = detect.detect_objects(images)
    return result
