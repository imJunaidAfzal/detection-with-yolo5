import shutil
from fastapi import FastAPI, UploadFile, File
from checking_yolo5 import DetectObject

app = FastAPI()


@app.post("/upload_file/")
async def create_upload_file(file: UploadFile = File(...)):
    detect = DetectObject()
    with open(f'{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = detect.detect_objects(file.filename)
    return result

