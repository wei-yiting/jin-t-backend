import os
from enum import Enum
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from zhconv import convert  # type: ignore

from app.config import (
    PUNC_FIX_MODEL_NAME,
    PUNC_FIX_PROMPT_FILE_PATH,
    PUNC_FIX_TEMPERATURE,
    TRANSCRIBE_PROMPT_FILE_PATH,
)
from app.utils import (
    add_spacing_between_chinese_english,
    check_transcript_punctuation_health,
    read_prompt,
)

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
    transcript_prompt = read_prompt(
        os.path.join(os.path.dirname(__file__), TRANSCRIBE_PROMPT_FILE_PATH)
    )
    response_from_transcribe = await client.audio.transcriptions.create(
        model=model_name.value,
        file=(audio_file.filename, audio_file.file),
        prompt=transcript_prompt,
    )
    raw_transcript = response_from_transcribe.text

    # 2. Convert simplified Chinese to traditional Chinese if any
    post_processed_transcript = convert(raw_transcript, "zh-tw")

    # 3. Add spacing between Chinese and English/numbers/English punctuation marks
    post_processed_transcript = add_spacing_between_chinese_english(
        post_processed_transcript
    )

    print(f"Post-processed transcript: {post_processed_transcript}")

    # 4. Check if the punctuation is healthy, if not, fix it with GPT-4o-mini
    if not check_transcript_punctuation_health(post_processed_transcript):
        print(
            "********* Punctuation is unhealthy, fixing with GPT-4o-mini... *********"
        )

        punc_fix_instruction_prompt = read_prompt(
            os.path.join(os.path.dirname(__file__), PUNC_FIX_PROMPT_FILE_PATH)
        )
        response_with_punctuation_fix = await client.responses.create(
            model=PUNC_FIX_MODEL_NAME,
            instructions=punc_fix_instruction_prompt,
            input=post_processed_transcript,
            temperature=PUNC_FIX_TEMPERATURE,
        )

        post_processed_transcript = response_with_punctuation_fix.output_text
        print(f"Fixed transcript: {post_processed_transcript}")

    return {"transcript": post_processed_transcript}


@app.api_route("/livez", methods=["GET", "HEAD"])
async def check_livez():
    return {"status": "live"}
