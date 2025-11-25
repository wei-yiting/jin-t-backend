from langsmith import traceable
from typing import Annotated
from fastapi import UploadFile, Depends
from openai import AsyncOpenAI
from langsmith.wrappers import wrap_openai

from app.config import (
    TRANSCRIBE_MODEL_NAME,
    PUNC_FIX_MODEL_NAME,
    REFINE_MODEL_NAME,
    PUNC_FIX_TEMPERATURE,
    REFINE_TEMPERATURE,
    PUNC_FIX_PROMPT_FILE_PATH,
    TRANSCRIBE_PROMPT_FILE_PATH,
    TRANSCRIPT_REFINE_PROMPT_FILE_PATH,
)
from app.lib.prompt_utils import read_prompt


@traceable(run_type="llm", name="LLM1_Transcribe")
async def transcribe_audio_to_text(
    audio_file: UploadFile,
    client: Annotated[AsyncOpenAI, Depends(wrap_openai)],
):
    transcript_prompt = read_prompt(TRANSCRIBE_PROMPT_FILE_PATH)
    return await client.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL_NAME,
        file=(audio_file.filename, audio_file.file),
        prompt=transcript_prompt,
    )


@traceable(run_type="llm", name="LLM2a_Fix_Punctuation")
async def fix_punctuation(
    transcript: str,
    client: Annotated[AsyncOpenAI, Depends(wrap_openai)],
):
    punc_fix_instruction_prompt = read_prompt(PUNC_FIX_PROMPT_FILE_PATH)
    return await client.responses.create(
        model=PUNC_FIX_MODEL_NAME,
        instructions=punc_fix_instruction_prompt,
        input=transcript,
        temperature=PUNC_FIX_TEMPERATURE,
    )


@traceable(run_type="llm", name="LLM2b_Refine_Transcript")
async def refine_transcript(
    transcript: str,
    client: Annotated[AsyncOpenAI, Depends(wrap_openai)],
):
    refine_prompt = read_prompt(TRANSCRIPT_REFINE_PROMPT_FILE_PATH)
    return await client.responses.create(
        model=REFINE_MODEL_NAME,
        instructions=refine_prompt,
        input=transcript,
        temperature=REFINE_TEMPERATURE,
    )
