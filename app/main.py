from enum import Enum
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

app = FastAPI()

# TODO: add audio file size validation (error handling)

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
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
    translation = await client.audio.transcriptions.create(
        model=model_name.value,
        file=(audio_file.filename, audio_file.file),
        prompt="""語音中的中文語句可能夾雜英文，保留英文部分不需翻譯直接轉換為英文文字，中文部分用台灣使用的繁體中文呈現。
        中英交錯時英文開始前與結束後應各加上一個空格，例如“這是 Speech to Text 工具”。
        依據語氣和語句順暢度，使用合適的標點符號。""",
    )

    return {"transcription": translation.text}
