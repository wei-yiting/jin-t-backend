import os
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, AuthenticationError, BadRequestError
from app.services.pipeline import run_transcription_pipeline
from app.models import (
    TranscribeMode,
    CheckIsOpenaiApiKeyValidResponse,
    CheckIsOpenaiApiKeyValidRequest,
)
from app.config import (
    MAX_FILE_SIZE_MB,
    FREE_TIER_MAX_AUDIO_DURATION_SECONDS,
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


@app.get("/")
async def root():
    return {"message": "This is the root endpoint of Jin-T Backend"}


@app.post("/transcribe")
async def convert_audio_to_text(
    audio_file: Annotated[UploadFile, File()],
    transcribe_mode: Annotated[TranscribeMode, Query(alias="mode")] = TranscribeMode.STANDARD,
    device_id: Annotated[str | None, Header(alias="X-Device-Id")] = None,
    consent_data_collection: Annotated[str | None, Header(alias="X-Consent-Data-Collection")] = None,
    personal_openai_api_key: Annotated[str | None, Header(alias="X-Custom-Openai-Api-Key")] = None,
    audio_duration: Annotated[str | None, Header(alias="X-Audio-Duration")] = None,
):  
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

    # Step 1: File size validation (all users) - 25MB limit
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB}MB."
        )

    # Step 2: API key and consent validation
    has_personal_openai_api_key = personal_openai_api_key and personal_openai_api_key.strip() != ""
    has_consent = consent_data_collection and consent_data_collection.lower() == "true"

    # Determine if using free tier
    if not has_personal_openai_api_key and not has_consent:
        # Scenario 1: No API key AND no consent
        raise HTTPException(
            status_code=403,
            detail="When not providing a custom API key, you must consent to data collection. If you do not consent to data collection, please provide your own OpenAI API key."
        )
    
    if has_personal_openai_api_key:
        # Scenario 2: Custom API key provided
        using_free_tier = False
        api_key_to_use = personal_openai_api_key
    else:
        # Scenario 3: No API key AND consent given (free tier)
        using_free_tier = True
        api_key_to_use = os.getenv("FREE_TIER_OPENAI_API_KEY")
        if not api_key_to_use:
            raise HTTPException(
                status_code=500,
                detail="Free tier OpenAI API key is not configured on the server."
            )

    # Step 3: Free tier audio duration validation
    if using_free_tier and audio_duration:
        try:
            duration_seconds = float(audio_duration)
            if duration_seconds > FREE_TIER_MAX_AUDIO_DURATION_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio duration exceeds the maximum limit of {FREE_TIER_MAX_AUDIO_DURATION_SECONDS // 60} minutes for free tier usage."
                )
        except ValueError:
            # Invalid duration format, continue without validation
            pass

    # Reset file pointer for processing
    await audio_file.seek(0)

    try:
        client = AsyncOpenAI(api_key=api_key_to_use)
        result = await run_transcription_pipeline(
            audio_file=audio_file,
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
        print(f"Unexpected error in transcription: {e}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during transcription"
        )


@app.post("/check-openai-api-key")
async def check_openai_api_key(
    request: CheckIsOpenaiApiKeyValidRequest,
) -> CheckIsOpenaiApiKeyValidResponse:
    """Checks if an OpenAI API key is valid by attempting to list models."""
    client = AsyncOpenAI(api_key=request.openai_api_key)
    try:
        await client.models.list()
        return CheckIsOpenaiApiKeyValidResponse(
            is_api_key_valid=True,
            has_unexpectied_validation_error=False,
        )
    except AuthenticationError:
        # AuthenticationError is raised when the API key is invalid
        return CheckIsOpenaiApiKeyValidResponse(
            is_api_key_valid=False,
            has_unexpectied_validation_error=False,
        )
    except Exception as e:
        # Handle other potential errors, e.g., network issues
        print(f"An unexpected error occurred when checking OpenAI API key: {e}")
        return CheckIsOpenaiApiKeyValidResponse(
            is_api_key_valid=False,
            has_unexpectied_validation_error=True,
        )


@app.api_route("/livez", methods=["GET", "HEAD"])
async def check_livez():
    return {"status": "live"}
