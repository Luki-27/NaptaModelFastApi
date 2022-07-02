import shutil
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from main import run
app = FastAPI()


@app.post("/")
def root(file: UploadFile= File(...)):

    with open(f'images\\img\\{file.filename}', "wb")as buffer:
        shutil.copyfileobj(file.file, buffer)

    return{"file_name": file.filename}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile= File(...)):
    return {"filename": file.filename}

@app.get("/")
def returnPredict():
     return {"name":run()[0]}


