from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Depends
from openai import AsyncOpenAI, AuthenticationError, BadRequestError
from app.services.pipeline import run_transcribe_pipeline
from app.models import TranscribeMode
from app.dependencies import (
    validate_file_size,
    get_api_key_config,
    check_and_update_rate_limit,
    validate_audio_duration,
    ApiKeyConfig,
    ValidatedAudioFile,
)

router = APIRouter()


@router.post("/transcribe")
async def convert_audio_to_text(
    validated_audio: Annotated[ValidatedAudioFile, Depends(validate_file_size)],
    api_key_config: Annotated[ApiKeyConfig, Depends(get_api_key_config)],
    transcribe_mode: Annotated[TranscribeMode, Query(alias="mode")],
    device_id: Annotated[str, Depends(check_and_update_rate_limit)],
    audio_duration: Annotated[str, Depends(validate_audio_duration)],
):
    try:
        client = AsyncOpenAI(api_key=api_key_config.api_key_to_use)
        result = await run_transcribe_pipeline(
            audio_file=validated_audio.file,
            llm_client=client,
            transcribe_mode=transcribe_mode,
            audio_duration=audio_duration,
        )
        return {"transcript": result}
    except BadRequestError as e:
        # OpenAI API returned 400 (corrupted/unsupported audio file)
        error_message = str(e)
        if (
            "corrupted" in error_message.lower()
            or "unsupported" in error_message.lower()
        ):
            raise HTTPException(
                status_code=400,
                detail="Audio file is corrupted or in an unsupported format",
            )
        raise HTTPException(status_code=400, detail=error_message)
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
    except Exception as e:
        # Log the error for debugging
        print(f"Unexpected error in transcribe pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during transcribe pipeline",
        )
