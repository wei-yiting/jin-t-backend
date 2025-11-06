import os
from enum import Enum
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from openai import AsyncOpenAI

load_dotenv()

app = FastAPI()

# TODO: make openai api key required
# TODO: add audio file size validation (error handling)


class ModelName(str, Enum):
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_TRANSCRIBE_DIARIZE = "gpt-4o-transcribe-diarize"


@app.post("/transcribe")
async def convert_audio_to_text(
    audio_file: Annotated[UploadFile, File()],
    model_name: Annotated[ModelName, Form()] = ModelName.GPT_4O_MINI_TRANSCRIBE,
    openai_api_key: Annotated[str | None, Form()] = os.getenv("OPENAI_API_KEY"),
):
    client = AsyncOpenAI(api_key=openai_api_key)
    translation = await client.audio.transcriptions.create(
        model=model_name.value,
        file=(audio_file.filename, audio_file.file),
    )

    return {"transcription": translation.text}
