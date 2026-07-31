import os
import asyncio
import re
from functools import partial
from io import BytesIO
import aiofiles
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from typing import cast, Callable, Awaitable
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langsmith.run_helpers import get_current_run_tree
from openai import AsyncOpenAI, APIConnectionError, RateLimitError

from app.config import (
    PUNC_FIX_MODEL_NAME,
    REFINE_MODEL_NAME,
    TEMP_AUDIO_FILES_DIR,
)
from app.models import (
    TranscribeStreamEventType,
    TranscribedResultChunk,
)
from app.lib.audio_tools import run_ffmpeg_to_slice_audio_file
from app.lib.transcript_processor import (
    check_transcript_punctuation_health,
    convert_simplified_to_traditional,
    add_spacing_between_chinese_english,
)
from app.services.llm_client import (
    transcribe_audio_to_text,
    fix_punctuation,
    refine_transcript,
    consolidate_chunks_text,
)


EventCallback = Callable[[TranscribeStreamEventType, dict], Awaitable[None]]

# Matches transcripts that are only markdown code fences and whitespace
_EMPTY_FENCE_PATTERN = re.compile(r"(?:\s|```[a-zA-Z]*)+")

async def no_op_event_callback(
    event_type: TranscribeStreamEventType, payload: dict
):
    """No-op event callback that does nothing."""
    pass


class _AudioFileWithName:
    """File-like wrapper carrying a filename attribute, required by the OpenAI audio API."""

    def __init__(self, file_obj: BytesIO, filename: str):
        self.file = file_obj
        self.filename = filename

class TranscribePipeline:
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        task_id: str,
        emit_event_callback: EventCallback = no_op_event_callback,
    ):
        self.llm_client = wrap_openai(llm_client)
        self.task_id = task_id
        self.emit_event_callback = emit_event_callback
        
        try:
            self.run_tree = get_current_run_tree()
        except Exception:
            self.run_tree = None


    def _add_langsmith_metadata_if_trancing_enabled(
        self,
        new_metadata: dict[str, str | float | bool | None],
    ):
        if self.run_tree:
            self.run_tree.add_metadata(new_metadata)


    async def _transcribe_audio_file(
        self, audio_file_path: str, error_label: str, remove_after_read: bool = False
    ) -> str:
        """Read an audio file into memory and transcribe it."""
        try:
            async with aiofiles.open(audio_file_path, "rb") as f:
                file_content = await f.read()

            audio_file = _AudioFileWithName(BytesIO(file_content), audio_file_path)
            response_from_transcribe = await transcribe_audio_to_text(
                audio_file, self.llm_client
            )
            return response_from_transcribe.text
        except Exception as e:
            raise Exception(f"Failed to transcribe {error_label}: {e}")
        finally:
            if remove_after_read and os.path.exists(audio_file_path):
                os.remove(audio_file_path)

    async def _post_process_and_emit(
        self, raw_transcript: str, chunk_index: int
    ) -> TranscribedResultChunk:
        """Normalize a raw transcript, record it, and stream it to the client."""
        code_post_processed_transcript = convert_simplified_to_traditional(
            raw_transcript
        )
        code_post_processed_transcript = add_spacing_between_chinese_english(
            code_post_processed_transcript
        )
        # The transcribe model sometimes wraps a no-speech response in an
        # empty markdown fence (```plaintext ... ```) instead of returning
        # the empty string the prompt asks for; normalize before the guard.
        if _EMPTY_FENCE_PATTERN.fullmatch(code_post_processed_transcript):
            code_post_processed_transcript = ""
        if not code_post_processed_transcript.strip():
            code_post_processed_transcript = ""
            self._add_langsmith_metadata_if_trancing_enabled(
                {f"chunk_{chunk_index}_silence_or_no_speech": True}
            )
        self._add_langsmith_metadata_if_trancing_enabled(
            {
                f"chunk_{chunk_index}_transcript": code_post_processed_transcript,
            }
        )

        transcribed_result_chunk = TranscribedResultChunk(
            chunk_index=chunk_index, text=code_post_processed_transcript
        )
        await self.emit_event_callback(
            TranscribeStreamEventType.CHUNK_COMPLETED,
            cast(dict, transcribed_result_chunk),
        )

        return transcribed_result_chunk

    @traceable(run_type="chain", name="Transcribe_Post_process_Whole_File")
    @retry(
        retry=retry_if_exception_type((APIConnectionError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def transcribe_whole_file(self, source_file_path: str) -> TranscribedResultChunk:
        """Transcribe -> Post-process -> Return stream content.

        Short-audio path: no slicing, so the source file is left for the
        caller to clean up.
        """
        raw_transcript = await self._transcribe_audio_file(
            source_file_path, error_label="audio file"
        )
        return await self._post_process_and_emit(raw_transcript, chunk_index=0)

    @traceable(run_type="chain", name="Slice_Transcribe_Post_process_Single_Chunk")
    @retry(
        retry=retry_if_exception_type((APIConnectionError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def slice_chunk_and_transcribe(self, chunk_meta: dict, source_file_path: str):
        """Slice -> Transcribe -> Post-process -> Return stream content"""
        chunk_index = chunk_meta["index"]
        chunk_file_path = f"{TEMP_AUDIO_FILES_DIR}/{self.task_id}_{chunk_index}.mp3"
        chunk_start_ms = chunk_meta["start_ms"]
        chunk_end_ms = chunk_meta["end_ms"]

        # 1. Slice the audio file
        try:
            loop = asyncio.get_running_loop()
            slice_audio_task = partial(
                run_ffmpeg_to_slice_audio_file,
                source_file_path,
                chunk_start_ms,
                chunk_end_ms,
                chunk_file_path,
            )
            await loop.run_in_executor(None, slice_audio_task)
        except Exception as e:
            if os.path.exists(chunk_file_path):
                os.remove(chunk_file_path)
            raise Exception(f"Failed to slice audio file: {e}")

        # 2. Transcribe the chunk
        raw_transcript = await self._transcribe_audio_file(
            chunk_file_path,
            error_label=f"audio chunk file index {chunk_index}",
            remove_after_read=True,
        )

        # 3. Post-process, emit and return
        return await self._post_process_and_emit(raw_transcript, chunk_index)


    @traceable(run_type="chain", name="Consolidate_Chunks_Text")
    async def consolidate_chunks_text(self, chunks_text: list[TranscribedResultChunk]) -> str:
        """Consolidate chunks text into a single transcript"""
        
        # 1. Format input string into a single string with tags:
        # <chunk index="0">...</chunk>
        # <chunk index="1">...</chunk>
        # ...
        sorted_chunks = sorted(chunks_text, key=lambda x: x["chunk_index"])

        # All chunks silent: skip the LLM call — consolidating empty input
        # risks the hallucination PR #6 guards against.
        if all(not chunk["text"].strip() for chunk in sorted_chunks):
            self._add_langsmith_metadata_if_trancing_enabled(
                {"silence_or_no_speech": True}
            )
            return ""

        chunks_with_tags = [
            f'<chunk index="{chunk["chunk_index"]}">{chunk["text"]}</chunk>'
            for chunk in sorted_chunks
        ]
        chunks_text_input_str = "\n".join(chunks_with_tags)
        self._add_langsmith_metadata_if_trancing_enabled({"chunk_merged_input": chunks_text_input_str})

        # 2. Consolidate chunks text
        consolidation_response = await consolidate_chunks_text(
            chunks_text_input_str, self.llm_client
        )
        
        # 3. Post-process the consolidated transcript
        post_processed_consolidated_transcript = add_spacing_between_chinese_english(consolidation_response.output_text)
        
        # 4. Add metadata to the run tree and return the post-processed consolidated transcript
        self._add_langsmith_metadata_if_trancing_enabled({"consolidation_response": post_processed_consolidated_transcript})
        return post_processed_consolidated_transcript

    
    @traceable(run_type="chain", name="Check_and_Fix_Punctuation")
    async def check_and_fix_punctuation(self, full_transcript: str) -> str:
        """Check and fix punctuation"""
        if check_transcript_punctuation_health(full_transcript):
            self._add_langsmith_metadata_if_trancing_enabled(
                {"has_punctuation_fixed": False}
            )
            return full_transcript
        await self.emit_event_callback(
            TranscribeStreamEventType.PUNC_FIXING,
            {"consolidated_text": full_transcript},
        )
        response_with_punctuation_fix = await fix_punctuation(
            full_transcript, self.llm_client
        )
        punctuation_fixed_transcript = response_with_punctuation_fix.output_text
        self._add_langsmith_metadata_if_trancing_enabled(
            {
                "has_punctuation_fixed": True,
                "punctuation_fixed_transcript": punctuation_fixed_transcript,
                "punc_fix_model_name": PUNC_FIX_MODEL_NAME,
            }
        )
        return punctuation_fixed_transcript


    @traceable(run_type="chain", name="Refine_Transcript")
    async def refine_transcript(self, full_transcript: str) -> str:
        """Refine the consolidated transcript (REFINED mode)"""
        await self.emit_event_callback(
            TranscribeStreamEventType.REFINING,
            {"consolidated_text": full_transcript},
        )
        response_from_refine = await refine_transcript(
            full_transcript, self.llm_client
        )
        refined_transcript = response_from_refine.output_text
        self._add_langsmith_metadata_if_trancing_enabled(
            {
                "refined_transcript": refined_transcript,
                "refine_model_name": REFINE_MODEL_NAME,
            }
        )
        return refined_transcript
