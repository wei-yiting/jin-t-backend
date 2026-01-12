from enum import Enum
from typing import TypedDict
from dataclasses import dataclass

from pydantic import BaseModel, Field
from fastapi import UploadFile

class TranscribeMode(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    REFINED = "refined"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskProcessingProgressCode(str, Enum):
    TRANSCRIBING = "transcribing"
    PUNC_FIXING = "punc_fixing"
    REFINING = "refining"


class TaskProgressResponse(BaseModel):
    status: TaskStatus
    progress_code: TaskProcessingProgressCode | None = None
    message: str
    transcript: str | None = None
    error_detail: str | None = None


class TranscribeRequestMetadata(TypedDict, total=False):
    device_id: str
    request_id: str
    using_personal_api_key: bool


class AudioMetadataFromRequest(TypedDict, total=False):
    audio_file_content_type: str | None
    audio_file_size_mb: float


class LangsmithRunTreeMetadata(
    AudioMetadataFromRequest, TranscribeRequestMetadata, total=False
):
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
    r2_object_key: str


class CheckIsOpenaiApiKeyValidRequest(BaseModel):
    encrypted_openai_api_key: str = Field(description="base64 encoded encrypted OpenAI API key")


class CheckIsOpenaiApiKeyValidResponse(BaseModel):
    is_api_key_valid: bool
    has_unexpectied_validation_error: bool


class UsageConfig(BaseModel):
    api_key_to_use: str
    using_free_tier: bool
    consent_data_collection: bool

    class Config:
        # Exclude from OpenAPI schema since this is internal only and contains sensitive information
        json_schema_extra = {"exclude": True}


class ValidatedAudioFile(BaseModel):
    file: UploadFile
    file_size_bytes: int
    file_size_mb: float

    class Config:
        # Exclude from OpenAPI schema since UploadFile is not serializable
        json_schema_extra = {"exclude": True}
        arbitrary_types_allowed = True

@dataclass(frozen=True, slots=True)
class RateLimitRule:
    key: str
    window_ttl: int
    max_duration: int | None = None 
    max_count: int | None = None
    error_duration_msg: str = "分鐘的限制"
    error_count_msg: str = "次轉錄的限制"
