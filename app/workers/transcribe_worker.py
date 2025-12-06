from redis.asyncio import Redis
from openai import AsyncOpenAI, AuthenticationError, BadRequestError
from app.models import (
    TranscribeMode,
    TranscribeRequestMetadata,
    TaskProcessingProgressCode,
)
from app.services.transcribe_pipeline import run_transcribe_pipeline
from app.services.task_progress import TaskProgressService


async def transcribe_worker(
    task_id: str,
    file_content: bytes,
    filename: str,
    openai_api_key: str,
    transcribe_mode: TranscribeMode,
    audio_duration: str | None,
    r2_object_key: str | None,
    transcribe_request_metadata: TranscribeRequestMetadata,
    redis: Redis,
):
    """
    Orchestrate the transcribe pipeline and task progress service.
    """
    task_progress_service = TaskProgressService(redis)

    # Define callback: when pipeline reports, update Redis
    async def pipeline_progress_reporter(
        progress_code: TaskProcessingProgressCode, message: str
    ):
        # This is where the pipeline "Status 1... Status 2..." is called
        await task_progress_service.update_progress(
            task_id,
            progress_code,
            message,
        )

    try:
        # Create client (inside worker to ensure thread safety)
        llm_client = AsyncOpenAI(api_key=openai_api_key)

        # Call core pipeline
        result = await run_transcribe_pipeline(
            file_content=file_content,  # use bytes type instead of UploadFile type from FastAPI for thread safety
            filename=filename,
            llm_client=llm_client,
            transcribe_mode=transcribe_mode,
            audio_duration=audio_duration,
            r2_object_key=r2_object_key,
            transcribe_request_metadata=transcribe_request_metadata,
            status_callback=pipeline_progress_reporter,  # inject the callback function to the pipeline
        )

        await task_progress_service.mark_completed(task_id, result)

    except BadRequestError as e:
        # OpenAI API returned 400 (corrupted/unsupported audio file)
        error_message = str(e)
        if (
            "corrupted" in error_message.lower()
            or "unsupported" in error_message.lower()
        ):
            error_message = "音檔損壞或音檔是不支援的格式"
        await task_progress_service.mark_failed(task_id, error_message)

    except AuthenticationError:
        await task_progress_service.mark_failed(
            task_id, "無效的 OpenAI API key 導致轉錄失敗"
        )

    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        await task_progress_service.mark_failed(task_id, str(e))
