import os
import shutil
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

