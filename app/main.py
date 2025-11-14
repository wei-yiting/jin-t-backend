from enum import Enum
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from utils.string_operation import add_spacing_between_chinese_english

from app.prompts import TRANSCRIBE_PROMPT

app = FastAPI()

# TODO: add audio file size validation (error handling)

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://jin-t-frontend.vercel.app",
    "https://jin-t.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ModelName(str, Enum):
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"


@app.get("/")
async def root():
    return {"message": "This is the root endpoint of Jin-T Backend"}


@app.post("/transcribe")
async def convert_audio_to_text(
    audio_file: Annotated[UploadFile, File()],
    openai_api_key: Annotated[str, Form()],
    model_name: Annotated[ModelName, Form()],
):
    client = AsyncOpenAI(api_key=openai_api_key)
    raw_transcription = await client.audio.transcriptions.create(
        model=model_name.value,
        file=(audio_file.filename, audio_file.file),
        prompt=TRANSCRIBE_PROMPT,
    )

    final_transcription = add_spacing_between_chinese_english(raw_transcription.text)

    return {"transcription": final_transcription}


@app.api_route("/health-check", methods=["GET", "HEAD"])
async def health_check():
    return {"message": "OK"}
