from typing import Annotated
import uuid

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Request
from redis.asyncio import Redis
from langsmith import tracing_context

from app.workers.transcribe_worker import transcribe_worker
from app.services.audio_storage import audio_storage_service
from app.models import (
    TranscribeMode,
    UsageConfig,
    ValidatedAudioFile,
    TranscribeRequestMetadata,
    TaskProgressResponse,
    TaskStatus,
    TaskProcessingProgressCode,
)
from app.services.task_progress import TaskProgressService
from app.dependencies import (
    validate_file_size,
    get_usage_config,
    check_and_update_rate_limit,
    validate_audio_duration,
    get_redis_client,
)

router = APIRouter(
    prefix="/transcribe",
    tags=["transcribe"],
)


@router.post("/start-task")
async def start_transcribe_task(
    validated_audio: Annotated[ValidatedAudioFile, Depends(validate_file_size)],
    usage_config: Annotated[UsageConfig, Depends(get_usage_config)],
    transcribe_mode: Annotated[TranscribeMode, Query(alias="mode")],
    device_id: Annotated[str, Depends(check_and_update_rate_limit)],
    audio_duration: Annotated[str, Depends(validate_audio_duration)],
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict[str, str]:
    request_id = str(uuid.uuid4())

    try:
        # 1. Read audio file to bytes to avoid the UploadFile being deleted after the request is completed
        #    so file content can be used in the background task even after this POST request is completed
        audio_bytes = await validated_audio.file.read()
        original_audio_filename = validated_audio.file.filename

        # 2. If consent data collection is enabled, upload the audio file to R2
        if usage_config.consent_data_collection:
            r2_object_key = audio_storage_service.generate_r2_object_key(
                device_id=device_id,
                request_id=request_id,
                filename=validated_audio.file.filename,
            )

            background_tasks.add_task(
                audio_storage_service.capture_raw_audio,
                file_content=audio_bytes,
                r2_object_key=r2_object_key,
                original_filename=original_audio_filename,
                content_type=validated_audio.file.content_type,
            )

            await validated_audio.file.seek(0)  # Reset file pointer to the beginning

        # 3. Initialize the task progress service
        redis_client = request.app.state.redis
        task_progress_service = TaskProgressService(redis_client)
        await task_progress_service.init_task(request_id)

        # 4. Start the transcribe worker in the background
        with tracing_context(enabled=usage_config.consent_data_collection):
            background_tasks.add_task(
                transcribe_worker,
                task_id=request_id,
                file_content=audio_bytes,
                filename=original_audio_filename or "",
                openai_api_key=usage_config.api_key_to_use,
                transcribe_mode=transcribe_mode,
                audio_duration=audio_duration,
                r2_object_key=r2_object_key or None,
                transcribe_request_metadata=TranscribeRequestMetadata(
                    device_id=device_id,
                    request_id=request_id,
                    using_personal_api_key=not usage_config.using_free_tier,
                ),
                redis=redis_client,
            )

        # 5. Return the task ID to the client to allow the client to poll the task progress
        return {"task_id": request_id}

    except Exception as e:
        # Log the error for debugging
        print(f"Unexpected error in transcribe process: {e}")
        raise HTTPException(
            status_code=500,
            detail="轉錄過程中發生意外錯誤",
        )


@router.get("/get-task-progress")
async def get_task_progress(
    task_id: str,
    redis_client: Annotated[Redis, Depends(get_redis_client)],
) -> TaskProgressResponse:
    task_progress_service = TaskProgressService(redis_client)
    try:
        task_progress = await task_progress_service.get_task_progress(task_id)

        if not task_progress:
            raise HTTPException(
                status_code=404,
                detail="transcribe task with ID {task_id} not found",
            )

        return TaskProgressResponse(
            status=TaskStatus(task_progress["status"]),
            progress_code=TaskProcessingProgressCode(task_progress["progress_code"]),
            message=task_progress["message"],
            transcript=task_progress["transcript"],
            error_detail=task_progress["error_detail"],
        )

    except Exception as e:
        print(f"Unexpected error in get task {task_id} progress from Redis: {e}")
        raise HTTPException(
            status_code=500,
            detail="取得轉錄進度與結果時發生意外錯誤",
        )
