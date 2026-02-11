from typing import cast, Callable, Awaitable
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langsmith.run_helpers import get_current_run_tree

from openai import AsyncOpenAI

from app.config import PUNC_FIX_MODEL_NAME, REFINE_MODEL_NAME, TRANSCRIBE_MODEL_NAME
from app.models import (
    LangsmithRunTreeMetadata,
    TranscribeMode,
    TranscribeRequestMetadata,
    TaskProcessingProgressCode,
)
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
from app.lib.uploadfile_memory import UploadFileInMemory

StatusCallback = Callable[[TaskProcessingProgressCode, str], Awaitable[None]]


async def no_op_status_callback(
    progress_code: TaskProcessingProgressCode, message: str
):
    """No-op status callback that does nothing."""
    pass


@traceable(run_type="chain", name="JinT_Main_Pipeline")
async def run_transcribe_pipeline(
    audio_file: UploadFileInMemory,
    llm_client: AsyncOpenAI,
    transcribe_mode: TranscribeMode,
    audio_duration: str | None,
    r2_object_key: str | None,
    transcribe_request_metadata: TranscribeRequestMetadata,
    status_callback: StatusCallback = no_op_status_callback,
):
    client = wrap_openai(llm_client)

    # If tracing is not enabled from tracing_context, run_tree will be None
    try:
        run_tree = get_current_run_tree()
    except Exception:
        run_tree = None

    def add_langsmith_metadata_if_trancing_enabled(
        new_meatadata: LangsmithRunTreeMetadata,
    ):
        if run_tree:
            run_tree.add_metadata(
                cast(dict[str, str | float | bool | None], new_meatadata)
            )

    # 0. Add already known metadata to the run tree
    audio_metadata = get_audio_metadata(audio_file)
    known_metadata: LangsmithRunTreeMetadata = {
        **audio_metadata,
        **transcribe_request_metadata,
        "transcribe_mode": transcribe_mode.value,
        "transcribe_model_name": TRANSCRIBE_MODEL_NAME,
        "audio_duration": float(audio_duration) if audio_duration else None,
    }

    if r2_object_key:
        known_metadata.update({"r2_object_key": r2_object_key})

    add_langsmith_metadata_if_trancing_enabled(known_metadata)

    # 1. Transcribe audio to text
    await status_callback(
        TaskProcessingProgressCode.TRANSCRIBING, "正在將語音轉換為文字..."
    )
    response_from_transcribe = await transcribe_audio_to_text(audio_file, client)
    raw_transcript = response_from_transcribe.text
    add_langsmith_metadata_if_trancing_enabled({"raw_transcript": raw_transcript})

    # 2. Post-process with code:
    # - Convert simplified Chinese to traditional Chinese if any
    # - Add spacing between Chinese and English/numbers/English punctuation marks
    code_post_processed_transcript = convert_simplified_to_traditional(raw_transcript)
    code_post_processed_transcript = add_spacing_between_chinese_english(
        code_post_processed_transcript
    )
    add_langsmith_metadata_if_trancing_enabled(
        {"code_post_processed_transcript": code_post_processed_transcript}
    )

    if not code_post_processed_transcript.strip():
        add_langsmith_metadata_if_trancing_enabled({"silence_or_no_speech": True})
        return ""

    # 3a. Transcribe mode: FAST -  return transcript with code post-processing
    if transcribe_mode == TranscribeMode.FAST:
        return code_post_processed_transcript

    # 3b. Transcribe mode: REFINED - Run LLM to refine the transcript
    if transcribe_mode == TranscribeMode.REFINED:
        await status_callback(TaskProcessingProgressCode.REFINING, "正在潤飾文字...")
        response_from_refine = await refine_transcript(
            code_post_processed_transcript, client
        )
        refined_transcript = response_from_refine.output_text
        add_langsmith_metadata_if_trancing_enabled(
            {
                "refined_transcript": refined_transcript,
                "refine_model_name": REFINE_MODEL_NAME,
            }
        )
        return refined_transcript

    # 3c. Transcribe mode: STANDARD - Check punctuation health to determine if LLM post-processing is needed
    if check_transcript_punctuation_health(code_post_processed_transcript):
        add_langsmith_metadata_if_trancing_enabled({"has_punctuation_fixed": False})
        return code_post_processed_transcript

    await status_callback(
        TaskProcessingProgressCode.PUNC_FIXING, "偵測到標點符號問題，正在修正..."
    )
    response_with_punctuation_fix = await fix_punctuation(
        code_post_processed_transcript, client
    )
    punctuation_fixed_transcript = response_with_punctuation_fix.output_text
    add_langsmith_metadata_if_trancing_enabled(
        {
            "has_punctuation_fixed": True,
            "punctuation_fixed_transcript": punctuation_fixed_transcript,
            "punc_fix_model_name": PUNC_FIX_MODEL_NAME,
        }
    )

    return punctuation_fixed_transcript
