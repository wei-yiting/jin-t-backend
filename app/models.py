from enum import Enum
from typing import TypedDict
from pydantic import BaseModel


class TranscribeMode(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    REFINED = "refined"


class AudioMetadataFromRequest(TypedDict):
    audio_file_content_type: str | None
    audio_file_size_mb: float


class LangsmithRunTreeMetadata(AudioMetadataFromRequest, total=False):
    transcribe_mode: str
    transcribe_model_name: str
    raw_transcript: str
    code_post_processed_transcript: str
    audio_duration: float | None
    has_punctuation_fixed: bool | None
    punctuation_fixed_transcript: str | None
    punc_fix_model_name: str | None
    refined_transcript: str | None
    refine_model_name: str | None


class CheckIsOpenaiApiKeyValidRequest(BaseModel):
    openai_api_key: str


class CheckIsOpenaiApiKeyValidResponse(BaseModel):
    is_api_key_valid: bool
    has_unexpectied_validation_error: bool
