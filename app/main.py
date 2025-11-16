from enum import Enum
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from zhconv import convert  # type: ignore

from app.prompts import TRANSCRIBE_PROMPT, TRANSCRIPT_PUNC_FIX_INSTRUCTION_PROMPT
from app.utils import (
    add_spacing_between_chinese_english,
    check_transcript_punctuation_health,
)

app = FastAPI()

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
    allow_methods=["GET", "HEAD", "POST"],
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

    # 1. Transcribe audio to text
    raw_transcript = await client.audio.transcriptions.create(
        model=model_name.value,
        file=(audio_file.filename, audio_file.file),
        prompt=TRANSCRIBE_PROMPT,
    )

    # 2. Convert simplified Chinese to traditional Chinese if any
    post_processed_transcript = convert(raw_transcript.text, "zh-tw")

    # 3. Add spacing between Chinese and English/numbers/English punctuation marks
    post_processed_transcript = add_spacing_between_chinese_english(
        post_processed_transcript
    )

    # 4. Check if the punctuation is healthy, if not, fix it with GPT-4o-mini
    if not check_transcript_punctuation_health(post_processed_transcript):
        print("Punctuation is unhealthy, fixing...")
        print(f"Original transcript: {post_processed_transcript}")
        response_with_punctuation_fix = await client.responses.create(
            model="gpt-4o-mini",
            instructions=TRANSCRIPT_PUNC_FIX_INSTRUCTION_PROMPT,
            input=post_processed_transcript,
            temperature=0.0,
        )

        post_processed_transcript = response_with_punctuation_fix.output_text
        print(f"Fixed transcript: {post_processed_transcript}")

    return {"transcript": post_processed_transcript}


@app.api_route("/livez", methods=["GET", "HEAD"])
async def check_livez():
    return {"status": "live"}
