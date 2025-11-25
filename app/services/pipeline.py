from typing import cast
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langsmith.run_helpers import get_current_run_tree
from fastapi import UploadFile

from openai import AsyncOpenAI

from app.config import PUNC_FIX_MODEL_NAME, REFINE_MODEL_NAME, TRANSCRIBE_MODEL_NAME
from app.models import LangsmithRunTreeMetadata, TranscribeMode
from app.lib.audio_tools import get_audio_metadata
from app.lib.transcript_processor import (
    check_transcript_punctuation_health,
    convert_simplified_to_traditional,
    add_spacing_between_chinese_english,
)
from app.services.llm_client import (
    transcribe_audio_to_text,
    fix_punctuation,
    refine_transcript,
)


@traceable(run_type="chain", name="JinT_Main_Pipeline")
async def run_transcription_pipeline(
    audio_file: UploadFile,
    llm_client: AsyncOpenAI,
    transcribe_mode: TranscribeMode,
    audio_duration: str | None,
):
    client = wrap_openai(llm_client)
    run_tree = get_current_run_tree()

    # 0. Get audio file metadata
    audio_metadata = get_audio_metadata(audio_file)
    run_tree_metadata: LangsmithRunTreeMetadata = {
        **audio_metadata,
        "transcribe_mode": transcribe_mode.value,
        "transcribe_model_name": TRANSCRIBE_MODEL_NAME,
        "audio_duration": float(audio_duration) if audio_duration else None,
    }

    # 1. Transcribe audio to text
    response_from_transcribe = await transcribe_audio_to_text(audio_file, client)
    raw_transcript = response_from_transcribe.text
    run_tree_metadata.update({"raw_transcript": raw_transcript})

    # 2. Post-process with code:
    # - Convert simplified Chinese to traditional Chinese if any
    # - Add spacing between Chinese and English/numbers/English punctuation marks
    code_post_processed_transcript = convert_simplified_to_traditional(raw_transcript)
    code_post_processed_transcript = add_spacing_between_chinese_english(
        code_post_processed_transcript
    )
    run_tree_metadata.update(
        {"code_post_processed_transcript": code_post_processed_transcript}
    )

    # 3a. Transcribe mode: FAST -  return transcript with code post-processing
    if transcribe_mode == TranscribeMode.FAST:
        if run_tree:
            run_tree.add_metadata(
                cast(dict[str, str | float | bool | None], run_tree_metadata)
            )
        return code_post_processed_transcript

    # 3b. Transcribe mode: REFINED - Run LLM to refine the transcript
    if transcribe_mode == TranscribeMode.REFINED:
        response_from_refine = await refine_transcript(
            code_post_processed_transcript, client
        )
        refined_transcript = response_from_refine.output_text
        run_tree_metadata.update(
            {
                "refined_transcript": refined_transcript,
                "refine_model_name": REFINE_MODEL_NAME,
            }
        )
        if run_tree:
            run_tree.add_metadata(
                cast(dict[str, str | float | bool | None], run_tree_metadata)
            )
        return refined_transcript

    # 3c. Transcribe mode: STANDARD - Check punctuation health to determine if LLM post-processing is needed
    if check_transcript_punctuation_health(code_post_processed_transcript):
        run_tree_metadata.update({"has_punctuation_fixed": False})
        if run_tree:
            run_tree.add_metadata(
                cast(dict[str, str | float | bool | None], run_tree_metadata)
            )
        return code_post_processed_transcript

    response_with_punctuation_fix = await fix_punctuation(
        code_post_processed_transcript, client
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
        run_tree.add_metadata(
            cast(dict[str, str | float | bool | None], run_tree_metadata)
        )

    return punctuation_fixed_transcript
