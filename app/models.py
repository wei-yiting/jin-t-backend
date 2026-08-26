from enum import Enum
from typing import TypedDict, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field
from fastapi import UploadFile

class TranscribeMode(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    REFINED = "refined"


class TranscribeStreamEventType(str, Enum):
    TASK_QUEUED = "TASK_QUEUED"  # Payload: {}
    TASK_STARTED = "TASK_STARTED"  # Payload: { total_chunks: int }
    CHUNK_COMPLETED = (
        "CHUNK_COMPLETED"  # Payload: TranscribedResultChunk { chunk_index: int, text: str }
    )
    CHUNKS_CONSOLIDATING = "CHUNKS_CONSOLIDATING"  # Payload: {}
    PUNC_FIXING = "PUNC_FIXING"  # Payload: { consolidated_text: str }
    REFINING = "REFINING"  # Payload: { consolidated_text: str }
    TASK_FINISHED = "TASK_FINISHED"  # Payload: { final_result: str }
    TASK_FAILED = "TASK_FAILED"  # Payload: { error: str }


class StreamEventResponse(BaseModel):
    messages: list[dict[str, Any]]
    last_id: str


class TranscribeRequestMetadata(TypedDict, total=False):
    device_id: str
    request_id: str
    using_personal_api_key: bool


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


class TranscribedResultChunk(TypedDict):
    chunk_index: int
    text: str
