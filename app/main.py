import os
from enum import Enum
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from app.services.pipeline import run_transcription_pipeline


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


class LLMModelName(str, Enum):
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"


@app.post("/transcribe")
async def convert_audio_to_text(
    audio_file: Annotated[UploadFile, File()],
    openai_api_key: Annotated[str, Form()],
    model_name: Annotated[LLMModelName, Form()],
):
    client = AsyncOpenAI(api_key=openai_api_key)
    result = await run_transcription_pipeline(audio_file, client, model_name.value)
    return {"transcript": result}


@app.api_route("/livez", methods=["GET", "HEAD"])
async def check_livez():
    return {"status": "live"}
