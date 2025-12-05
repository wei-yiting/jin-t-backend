from typing import Annotated
import uuid

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from openai import AsyncOpenAI, AuthenticationError, BadRequestError
from langsmith import tracing_context

from app.services.transcribe_pipeline import run_transcribe_pipeline
from app.services.audio_storage import audio_storage_service
from app.models import (
    TranscribeMode,
    UsageConfig,
    ValidatedAudioFile,
    TranscribeRequestMetadata,
)
from app.dependencies import (
    validate_file_size,
    get_usage_config,
    check_and_update_rate_limit,
    validate_audio_duration,
)

router = APIRouter()


@router.post("/transcribe")
async def convert_audio_to_text(
    validated_audio: Annotated[ValidatedAudioFile, Depends(validate_file_size)],
    usage_config: Annotated[UsageConfig, Depends(get_usage_config)],
    transcribe_mode: Annotated[TranscribeMode, Query(alias="mode")],
    device_id: Annotated[str, Depends(check_and_update_rate_limit)],
    audio_duration: Annotated[str, Depends(validate_audio_duration)],
    background_tasks: BackgroundTasks,
):
    try:
        request_id = str(uuid.uuid4())
        client = AsyncOpenAI(api_key=usage_config.api_key_to_use)

        if usage_config.consent_data_collection:
            r2_object_key = audio_storage_service.generate_r2_object_key(
                device_id=device_id,
                request_id=request_id,
                filename=validated_audio.file.filename,
            )

            file_bytes = await validated_audio.file.read()

            background_tasks.add_task(
                audio_storage_service.capture_raw_audio,
                file_content=file_bytes,
                r2_object_key=r2_object_key,
                original_filename=validated_audio.file.filename,
                content_type=validated_audio.file.content_type,
            )

            await validated_audio.file.seek(0)  # Reset file pointer to the beginning

        with tracing_context(enabled=usage_config.consent_data_collection):
            result = await run_transcribe_pipeline(
                audio_file=validated_audio.file,
                llm_client=client,
                transcribe_mode=transcribe_mode,
                audio_duration=audio_duration,
                r2_object_key=r2_object_key
                if usage_config.consent_data_collection
                else None,
                transcribe_request_metadata=TranscribeRequestMetadata(
                    device_id=device_id,
                    request_id=request_id,
                    using_personal_api_key=not usage_config.using_free_tier,
                ),
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
