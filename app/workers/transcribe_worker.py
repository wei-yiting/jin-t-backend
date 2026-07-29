import asyncio
import os

from redis.asyncio import Redis
from openai import AsyncOpenAI, AuthenticationError, BadRequestError

from app.config import MAX_CONCURRENT_TRANSCRIBE_WORKERS, INITIAL_CONCURRENT_CHUNK_DURATIONS_SECONDS, SUBSEQUENT_CHUNK_DURATION_SECONDS, AUDIO_CHUNK_OVERLAP_MS
from app.models import (
    TranscribeMode,
    TranscribeRequestMetadata,
    TranscribeStreamEventType,
    AudioChunkMetadata
)
from app.pipelines.transcribe_pipeline import TranscribePipeline
from app.services.transcribe_stream import TranscribeStreamService
from app.lib.audio_tools import run_ffmpeg_to_get_audio_duration


class TranscribeWorker:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIBE_WORKERS)
    
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
            # 1. calculate total duration and chunks
            total_duration_ms = run_ffmpeg_to_get_audio_duration(file_path)
            chunks = self._generate_audio_chunk_metadata(total_duration_ms)
            
            # 2. emit started event
            await self.emit_stream_event(TranscribeStreamEventType.TASK_STARTED, {
                "total_chunks": len(chunks)
            })
            
            # 3. execute chunks in parallel (Scatter)
            tasks = []
            for chunk in chunks:
                tasks.append(self._process_chunk_safe(transcribe_pipeline, chunk, file_path))
            
            transcribed_result_chunks = await asyncio.gather(*tasks)
            await self.emit_stream_event(TranscribeStreamEventType.CHUNKS_CONSOLIDATING, {})

            # 4. consolidate chunks text
            consolidated_text = await transcribe_pipeline.consolidate_chunks_text(transcribed_result_chunks)

            # 5. mode-specific post-processing on the consolidated transcript
            final_result = await self._post_process_by_mode(transcribe_pipeline, consolidated_text)
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

    @classmethod
    async def _process_chunk_safe(cls, pipeline, chunk, file_path):
        """
        Wrapper function to handle Semaphore
        """
        async with cls.semaphore:
            return await pipeline.slice_chunk_and_transcribe(chunk, file_path)


    async def emit_stream_event(self, event_type: TranscribeStreamEventType, payload: dict):
        await self.transcribe_stream_service.emit_event(self.task_id, event_type, payload)

    
    @staticmethod
    def _generate_audio_chunk_metadata(total_ms: int) -> list[AudioChunkMetadata]:
        """
        Generate audio chunk metadata for slicing the audio file
        """
        if total_ms <= 0:
            return []

        chunks: list[AudioChunkMetadata] = []
        start_ms = 0
        idx = 0

        while start_ms < total_ms:
            if idx < len(INITIAL_CONCURRENT_CHUNK_DURATIONS_SECONDS):
                current_chunk_duration_seconds = INITIAL_CONCURRENT_CHUNK_DURATIONS_SECONDS[idx]
            else:
                current_chunk_duration_seconds = SUBSEQUENT_CHUNK_DURATION_SECONDS

            end_ms = start_ms + current_chunk_duration_seconds * 1000
            end_ms = min(end_ms, total_ms)
            chunks.append(AudioChunkMetadata(index=idx, start_ms=start_ms, end_ms=end_ms))

            # If we've reached the end, break out
            if end_ms >= total_ms:
                break

            # Overlap 5秒：下一段從 (end - 5000ms) 開始
            start_ms = max(end_ms - AUDIO_CHUNK_OVERLAP_MS, 0)
            idx += 1

        return chunks