# services/pipeline.py
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langsmith.run_helpers import get_current_run_tree
from fastapi import UploadFile

from openai import AsyncOpenAI

from app.config import PUNC_FIX_MODEL_NAME
from app.lib.audio_tools import get_audio_metadata
from app.lib.transcript_processor import (
    check_transcript_punctuation_health,
    convert_simplified_to_traditional,
    add_spacing_between_chinese_english,
)
from app.services.llm_client import transcribe_audio_to_text, fix_punctuation


@traceable(run_type="chain", name="JinT_Main_Pipeline")
async def run_transcription_pipeline(
    audio_file: UploadFile,
    llm_client: AsyncOpenAI,
    transcribe_model_name: str,
    audio_duration: str | None,
):
    client = wrap_openai(llm_client)
    run_tree = get_current_run_tree()
    run_tree_metadata: dict[str, bool | str | None | float] = {}

    # 0. Get audio file metadata
    audio_metadata = get_audio_metadata(audio_file)
    audio_metadata.update(
        {"audio_duration": float(audio_duration) if audio_duration else None}
    )
    run_tree_metadata.update(audio_metadata)

    # 1. Transcribe audio to text
    response_from_transcribe = await transcribe_audio_to_text(
        audio_file, transcribe_model_name, client
    )
    raw_transcript = response_from_transcribe.text
    run_tree_metadata.update(
        {
            "raw_transcript": raw_transcript,
            "transcribe_model_name": transcribe_model_name,
        }
    )

    # 2. Convert simplified Chinese to traditional Chinese if any
    post_processed_transcript = convert_simplified_to_traditional(raw_transcript)

    # 3. Add spacing between Chinese and English/numbers/English punctuation marks
    post_processed_transcript = add_spacing_between_chinese_english(
        post_processed_transcript
    )

    run_tree_metadata.update({"post_processed_transcript": post_processed_transcript})

    # 4a. If the punctuation is healthy, add metadata and return the transcript
    if check_transcript_punctuation_health(post_processed_transcript):
        run_tree_metadata.update(
            {
                "has_punctuation_fixed": False,
                "punctuation_fixed_transcript": None,
                "punc_fix_model_name": None,
            }
        )
        if run_tree:
            run_tree.add_metadata(run_tree_metadata)
        return post_processed_transcript

    # 4b. Fix the punctuation with LLM, add metadata and return the transcript
    response_with_punctuation_fix = await fix_punctuation(
        post_processed_transcript, client
    )
    punctuation_fixed_transcript = response_with_punctuation_fix.output_text

    run_tree_metadata.update(
        {
            "has_punctuation_fixed": True,
            "punctuation_fixed_transcript": punctuation_fixed_transcript,
            "punc_fix_model_name": PUNC_FIX_MODEL_NAME,
        }
    )
    if run_tree:
        run_tree.add_metadata(run_tree_metadata)

    return punctuation_fixed_transcript
