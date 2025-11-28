import os
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, AuthenticationError
from app.services.pipeline import run_transcription_pipeline
from app.models import (
    TranscribeMode,
    CheckIsOpenaiApiKeyValidResponse,
    CheckIsOpenaiApiKeyValidRequest,
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
    openai_api_key: Annotated[str, Form()],
    transcribe_mode: Annotated[TranscribeMode, Form()],
    audio_duration: Annotated[str | None, Form()] = None,
):
    client = AsyncOpenAI(api_key=openai_api_key)
    result = await run_transcription_pipeline(
        audio_file=audio_file,
        llm_client=client,
        transcribe_mode=transcribe_mode,
        audio_duration=audio_duration,
    )
    return {"transcript": result}


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
