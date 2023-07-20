from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from io import BytesIO
import main
import json

app = FastAPI()

# Instead of allowing all origins, specify the trusted ones.
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://192.168.50.230:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionSettings(BaseModel):
    task: str
    min_speakers: int
    max_speakers: int
    whisper_model: str
    batch_size: int
    compute_type: str
    dump_model: bool

    class Config:
        schema_extra = {
            "example": {
                "task": "t",
                "min_speakers": 2,
                "max_speakers": 2,
                "whisper_model": "large-v2",
                "batch_size": 16,
                "compute_type": "float16",
                "dump_model": False
            }
        }

def validate_settings(settings: TranscriptionSettings):
    if settings.task not in ["t", "td"]:
        raise HTTPException(status_code=400, detail="Invalid task. Valid values are 't' or 'td'.")
    if settings.compute_type not in ["float16", "int8"]:
        raise HTTPException(status_code=400, detail="Invalid compute_type. Valid values are 'float16' or 'int8'.")
    if settings.whisper_model not in ["tiny", "base", "small", "medium", "large", "large-v2"]:
        raise HTTPException(status_code=400, detail="Invalid whisper_model.")
    if not (1 <= settings.min_speakers <= 10) or not (1 <= settings.max_speakers <= 10):
        raise HTTPException(status_code=400, detail="Invalid number of speakers. Value should be between 1 and 10.")
    if not (1 <= settings.batch_size <= 64):
        raise HTTPException(status_code=400, detail="Invalid batch size. Value should be between 1 and 64.")

@app.post("/process_audio")
async def process_audio(settings: TranscriptionSettings, file: UploadFile = Form(...)):
    validate_settings(settings)
    try:
        audio_file = BytesIO(await file.read())
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read the uploaded file.")

    try:
        result = main.main(settings.task, audio_file, main.device, settings.batch_size, settings.compute_type, settings.dump_model, settings.min_speakers, settings.max_speakers, settings.whisper_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
