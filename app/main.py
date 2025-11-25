import os
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from app.services.pipeline import run_transcription_pipeline
from app.models import TranscribeMode

app = FastAPI()

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [
    origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "HEAD", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "This is the root endpoint of Jin-T Backend"}


@app.post("/transcribe")
async def convert_audio_to_text(
    audio_file: Annotated[UploadFile, File()],
    openai_api_key: Annotated[str, Form()],
    transcribe_mode: Annotated[TranscribeMode, Form()],
    audio_duration: Annotated[str | None, Form()] = None,
):
    client = AsyncOpenAI(api_key=openai_api_key)
    result = await run_transcription_pipeline(
        audio_file=audio_file,
        llm_client=client,
        transcribe_mode=transcribe_mode,
        audio_duration=audio_duration,
    )
    return {"transcript": result}


@app.api_route("/livez", methods=["GET", "HEAD"])
async def check_livez():
    return {"status": "live"}
