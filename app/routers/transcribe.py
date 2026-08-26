from typing import Annotated
import json
import uuid

import aiofiles
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Request
from redis.asyncio import Redis
from langsmith import tracing_context

from app.workers.transcribe_worker import TranscribeWorker
from app.services.audio_storage import audio_storage_service
from app.models import (
    TranscribeMode,
    UsageConfig,
    ValidatedAudioFile,
    TranscribeRequestMetadata,
    StreamEventResponse,
)
from app.services.transcribe_stream import TranscribeStreamService
from app.dependencies import (
    validate_file_size,
    get_usage_config,
    check_and_update_rate_limit,
    validate_audio_duration,
    get_redis_client,
)
from app.config import TEMP_AUDIO_FILES_DIR
from app.lib.audio_tools import resolve_audio_file_extension

router = APIRouter(
    prefix="/transcribe-tasks",
    tags=["transcribe"],
)


@router.post("")
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

    # Validated before the try block so its 400 is not swallowed into a 500.
    # The stored file must keep the uploaded extension: short audio is sent to
    # the transcription API as-is, and that API reads the container format from
    # the filename — a browser `.webm` recording stored as `.mp3` is rejected
    # as corrupted.
    audio_file_extension = resolve_audio_file_extension(validated_audio.file.filename)

    try:
        # 1. Read audio file to bytes to avoid the UploadFile being deleted after the request is completed
        #    so file content can be used in the background task even after this POST request is completed
        stored_audio_file_path=f"{TEMP_AUDIO_FILES_DIR}/{request_id}{audio_file_extension}"
        async with aiofiles.open(stored_audio_file_path, 'wb') as stored_audio_file:
            while content := await validated_audio.file.read(1024 * 1024):
                await stored_audio_file.write(content)

        # 2. If consent data collection is enabled, upload the audio file to R2
        if usage_config.consent_data_collection:
            r2_object_key = audio_storage_service.generate_r2_object_key(
                device_id=device_id,
                request_id=request_id,
                filename=validated_audio.file.filename,
            )

            # Re-read from disk: the streamed upload has been consumed by the
            # chunked write above, so the file on disk is the only full copy.
            async with aiofiles.open(stored_audio_file_path, "rb") as stored_file:
                file_content = await stored_file.read()

            background_tasks.add_task(
                audio_storage_service.capture_raw_audio,
                file_content=file_content,
                r2_object_key=r2_object_key,
                original_filename=validated_audio.file.filename,
                content_type=validated_audio.file.content_type,
            )

        # 3. Initialize the transcribe stream service
        redis_client = request.app.state.redis
        transcribe_stream_service = TranscribeStreamService(redis_client)
        await transcribe_stream_service.init_task(request_id)

        # 4. Start the transcribe worker in the background
        transcribe_worker = TranscribeWorker(
            task_id=request_id,
            openai_api_key=usage_config.api_key_to_use,
            transcribe_mode=transcribe_mode,
            r2_object_key=r2_object_key if usage_config.consent_data_collection else None,
            transcribe_request_metadata=TranscribeRequestMetadata(device_id=device_id, request_id=request_id, using_personal_api_key=not usage_config.using_free_tier),
            redis=redis_client,
        )
        with tracing_context(enabled=usage_config.consent_data_collection):
            background_tasks.add_task(
                transcribe_worker.run,
                file_path=stored_audio_file_path,
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


@router.get("/{task_id}")
async def get_task_progress(
    task_id: str,
    redis_client: Annotated[Redis, Depends(get_redis_client)],
    last_id: Annotated[str, Query(description="Last received message ID in the stream")] = "0-0"
) -> StreamEventResponse:
    """
    Long polling to get the transcribe stream events.
    """
    transcribe_stream_service = TranscribeStreamService(redis_client)
    try:
        streams = await transcribe_stream_service.read_stream(task_id, last_id, 20 * 1000)
    except Exception as e:
        print(f"Unexpected error in get task {task_id} progress from Redis: {e}")
        raise HTTPException(
            status_code=500,
            detail="取得轉錄進度與結果時發生意外錯誤",
        )

    def _parse_payload(raw_payload: object) -> object:
        if isinstance(raw_payload, (bytes, bytearray)):
            try:
                raw_payload = raw_payload.decode("utf-8")
            except Exception:
                return raw_payload
        if isinstance(raw_payload, str):
            if raw_payload.strip() == "null":
                return None
            try:
                return json.loads(raw_payload)
            except json.JSONDecodeError:
                return raw_payload
        return raw_payload

    def _parse_event_type(raw_event_type: object) -> object:
        if isinstance(raw_event_type, (bytes, bytearray)):
            try:
                return raw_event_type.decode("utf-8")
            except Exception:
                return raw_event_type
        return raw_event_type

    messages: list[dict[str, object]] = []
    new_last_id = last_id

    if streams:
        message_list = streams[0][1]

        for msg in message_list:
            message_id = msg[0]
            message_data = msg[1]

            messages.append({
                "id": message_id,
                "type": _parse_event_type(message_data["event_type"]),
                "payload": _parse_payload(message_data["payload"]),
            })

            new_last_id = message_id

    return StreamEventResponse(messages=messages, last_id=new_last_id)
