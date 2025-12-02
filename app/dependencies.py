"""FastAPI dependencies for transcribe endpoint validation."""

import os
from typing import Annotated

from fastapi import Header, HTTPException, UploadFile, Depends, File
from pydantic import BaseModel
from app.config import MAX_FILE_SIZE_MB, FREE_TIER_MAX_AUDIO_DURATION_SECONDS


class ApiKeyConfig(BaseModel):
    """Configuration for API key usage."""

    api_key_to_use: str
    using_free_tier: bool

    class Config:
        # Exclude from OpenAPI schema since this is internal only
        # and contains sensitive information
        json_schema_extra = {"exclude": True}


class ValidatedAudioFile(BaseModel):
    """Validated audio file information."""

    file: UploadFile
    file_size_bytes: int
    file_size_mb: float

    class Config:
        # Exclude from OpenAPI schema since UploadFile is not serializable
        json_schema_extra = {"exclude": True}
        arbitrary_types_allowed = True


async def validate_audio_file(
    audio_file: Annotated[UploadFile, File()],
) -> ValidatedAudioFile:
    """Validate audio file filename and minimum size (100 bytes)."""
    # Validate audio file
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Check file size (must be > 100 bytes for valid audio)
    file_content = await audio_file.read()
    file_size_bytes = len(file_content)

    if file_size_bytes < 100:
        raise HTTPException(
            status_code=400, detail="Audio file is too small or corrupted"
        )

    file_size_mb = file_size_bytes / (1024 * 1024)

    return ValidatedAudioFile(
        file=audio_file, file_size_bytes=file_size_bytes, file_size_mb=file_size_mb
    )


async def validate_file_size(
    validated_audio: Annotated[ValidatedAudioFile, Depends(validate_audio_file)],
) -> ValidatedAudioFile:
    """Validate file size does not exceed 25MB limit."""
    if validated_audio.file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB}MB.",
        )

    # Reset file pointer for processing
    await validated_audio.file.seek(0)

    return validated_audio


def get_api_key_config(
    personal_openai_api_key: Annotated[str, Header(alias="X-Custom-Openai-Api-Key")],
    consent_data_collection: Annotated[str, Header(alias="X-Consent-Data-Collection")],
) -> ApiKeyConfig:
    """Determine API key configuration based on headers."""
    has_personal_openai_api_key = (
        personal_openai_api_key and personal_openai_api_key.strip() != ""
    )
    has_consent = consent_data_collection and consent_data_collection.lower() == "true"

    # Determine if using free tier
    if not has_personal_openai_api_key and not has_consent:
        # Scenario 1: No API key AND no consent
        raise HTTPException(
            status_code=403,
            detail="When not providing a custom API key, you must consent to data collection. If you do not consent to data collection, please provide your own OpenAI API key.",
        )

    if has_personal_openai_api_key:
        # Scenario 2: Custom API key provided
        using_free_tier = False
        api_key_to_use = personal_openai_api_key
    else:
        # Scenario 3: No API key AND consent given (free tier)
        using_free_tier = True
        api_key_to_use = os.getenv("FREE_TIER_OPENAI_API_KEY", "")
        if not api_key_to_use:
            raise HTTPException(
                status_code=500,
                detail="Free tier OpenAI API key is not configured on the server.",
            )

    return ApiKeyConfig(api_key_to_use=api_key_to_use, using_free_tier=using_free_tier)


def validate_audio_duration(
    audio_duration: Annotated[str, Header(alias="X-Audio-Duration")],
    api_key_config: Annotated[ApiKeyConfig, Depends(get_api_key_config)],
) -> str:
    """Validate free tier audio duration does not exceed 10 minutes."""
    if api_key_config.using_free_tier and audio_duration:
        try:
            duration_seconds = float(audio_duration)
            if duration_seconds > FREE_TIER_MAX_AUDIO_DURATION_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio duration exceeds the maximum limit of {FREE_TIER_MAX_AUDIO_DURATION_SECONDS // 60} minutes for free tier usage.",
                )
        except ValueError:
            # Invalid duration format, continue without validation
            pass

    return audio_duration
