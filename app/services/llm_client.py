import os
from langsmith import traceable
from typing import Annotated
from fastapi import UploadFile, Depends
from openai import AsyncOpenAI
from langsmith.wrappers import wrap_openai

from app.config import (
    PUNC_FIX_MODEL_NAME,
    PUNC_FIX_PROMPT_FILE_PATH,
    PUNC_FIX_TEMPERATURE,
    TRANSCRIBE_PROMPT_FILE_PATH,
)
from app.utils import read_prompt


@traceable(run_type="llm", name="LLM1_Transcribe")
async def transcribe_audio_to_text(
    audio_file: UploadFile,
    model_name: str,
    client: Annotated[AsyncOpenAI, Depends(wrap_openai)],
):
    transcript_prompt = read_prompt(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), TRANSCRIBE_PROMPT_FILE_PATH
        )
    )
    return await client.audio.transcriptions.create(
        model=model_name,
        file=(audio_file.filename, audio_file.file),
        prompt=transcript_prompt,
    )


@traceable(run_type="llm", name="LLM2_Fix_Punctuation")
async def fix_punctuation(
    transcript: str,
    client: Annotated[AsyncOpenAI, Depends(wrap_openai)],
):
    punc_fix_instruction_prompt = read_prompt(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), PUNC_FIX_PROMPT_FILE_PATH
        )
    )
    return await client.responses.create(
        model=PUNC_FIX_MODEL_NAME,
        instructions=punc_fix_instruction_prompt,
        input=transcript,
        temperature=PUNC_FIX_TEMPERATURE,
    )
