import os

from redis.asyncio import Redis
from openai import AsyncOpenAI, AuthenticationError, BadRequestError

from app.models import (
    TranscribeMode,
    TranscribeRequestMetadata,
    TranscribeStreamEventType,
)
from app.pipelines.transcribe_pipeline import TranscribePipeline
from app.services.transcribe_stream import TranscribeStreamService


class TranscribeWorker:
    def __init__(
        self,
        task_id: str,
        openai_api_key: str,
        transcribe_mode: TranscribeMode,
        r2_object_key: str | None,
        transcribe_request_metadata: TranscribeRequestMetadata,
        redis: Redis,
    ):
        self.task_id = task_id
        self.llm_client = AsyncOpenAI(api_key=openai_api_key)
        self.transcribe_mode = transcribe_mode
        self.r2_object_key = r2_object_key
        self.transcribe_request_metadata = transcribe_request_metadata
        self.transcribe_stream_service = TranscribeStreamService(redis)

    async def run(self, file_path: str):
        transcribe_pipeline = TranscribePipeline(self.llm_client, self.task_id, self.emit_stream_event)

        try:
            transcript = await self._run_single_call(transcribe_pipeline, file_path)

            # 2. mode-specific post-processing on the full transcript
            final_result = await self._post_process_by_mode(transcribe_pipeline, transcript)
            await self.emit_stream_event(TranscribeStreamEventType.TASK_FINISHED, {
                "final_result": final_result
            })

        except BadRequestError as e:
            error_message = str(e)
            if (
                "corrupted" in error_message.lower()
                or "unsupported" in error_message.lower()
            ):
                error_message = "音檔損壞或音檔是不支援的格式"
            await self.emit_stream_event(TranscribeStreamEventType.TASK_FAILED, {"error": error_message})

        except AuthenticationError:
            await self.emit_stream_event(
                TranscribeStreamEventType.TASK_FAILED,
                {"error": "無效的 OpenAI API key 導致轉錄失敗"},
            )

        except Exception as e:
            print(f"Task Failed: {e}")
            await self.emit_stream_event(TranscribeStreamEventType.TASK_FAILED, {"error": str(e)})

        finally:
            # [重要] 清理原始大檔
            if os.path.exists(file_path):
                os.remove(file_path)

    async def _run_single_call(
        self, pipeline: TranscribePipeline, file_path: str
    ) -> str:
        """Transcribe short audio in one call — no slicing, no consolidation."""
        await self.emit_stream_event(
            TranscribeStreamEventType.TASK_STARTED, {"total_chunks": 1}
        )
        result = await pipeline.transcribe_whole_file(file_path)
        return result["text"]

    async def _post_process_by_mode(
        self, pipeline: TranscribePipeline, consolidated_text: str
    ) -> str:
        """Apply the transcribe mode's post-processing to the consolidated transcript."""
        if not consolidated_text.strip():
            return ""
        if self.transcribe_mode == TranscribeMode.REFINED:
            return await pipeline.refine_transcript(consolidated_text)
        if self.transcribe_mode == TranscribeMode.STANDARD:
            return await pipeline.check_and_fix_punctuation(consolidated_text)
        return consolidated_text  # FAST: per-chunk code post-processing only

    async def emit_stream_event(self, event_type: TranscribeStreamEventType, payload: dict):
        await self.transcribe_stream_service.emit_event(self.task_id, event_type, payload)
