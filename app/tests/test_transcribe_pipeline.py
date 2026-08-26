"""Unit tests for the transcribe pipeline and worker (single-call path)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.config import SUPPORTED_AUDIO_EXTENSIONS
from app.models import TranscribeMode, TranscribeStreamEventType, TranscribedResultChunk
from app.pipelines.transcribe_pipeline import TranscribePipeline
from app.workers.transcribe_worker import TranscribeWorker

pytestmark = pytest.mark.anyio

PIPELINE_MODULE = "app.pipelines.transcribe_pipeline"


def _patch_langsmith():
    return patch(f"{PIPELINE_MODULE}.get_current_run_tree", side_effect=Exception)


def _patch_wrap_openai():
    return patch(f"{PIPELINE_MODULE}.wrap_openai", side_effect=lambda c: c)


def _make_pipeline(emit_callback=None):
    with _patch_langsmith(), _patch_wrap_openai():
        return TranscribePipeline(
            llm_client=AsyncMock(),
            task_id="test-task",
            emit_event_callback=emit_callback or AsyncMock(),
        )


def _make_worker(mode: TranscribeMode) -> TranscribeWorker:
    worker = TranscribeWorker(
        task_id="test-task",
        openai_api_key="test-key",
        transcribe_mode=mode,
        r2_object_key=None,
        transcribe_request_metadata={},
        redis=MagicMock(),
    )
    worker.transcribe_stream_service = MagicMock(emit_event=AsyncMock())
    return worker


class TestResolveAudioFileExtension:
    """The stored file must keep the uploaded extension — the transcription API
    reads the container format from the filename, so a `.webm` recording stored
    as `.mp3` comes back as 'corrupted or unsupported'."""

    def test_keeps_browser_recording_extension(self):
        from app.lib.audio_tools import resolve_audio_file_extension

        assert resolve_audio_file_extension("recording.webm") == ".webm"

    def test_normalizes_case(self):
        from app.lib.audio_tools import resolve_audio_file_extension

        assert resolve_audio_file_extension("Recording.MP3") == ".mp3"

    @pytest.mark.parametrize(
        "filename", ["recording.webm", "a.mp3", "b.m4a", "c.wav", "d.ogg", "e.mp4"]
    )
    def test_accepts_supported_formats(self, filename):
        from app.lib.audio_tools import resolve_audio_file_extension

        assert resolve_audio_file_extension(filename) in SUPPORTED_AUDIO_EXTENSIONS

    @pytest.mark.parametrize("filename", ["notes.txt", "noextension", "", None])
    def test_rejects_unsupported_or_missing(self, filename):
        from fastapi import HTTPException
        from app.lib.audio_tools import resolve_audio_file_extension

        with pytest.raises(HTTPException) as exc:
            resolve_audio_file_extension(filename)
        assert exc.value.status_code == 400

    def test_path_traversal_in_filename_is_rejected(self):
        from fastapi import HTTPException
        from app.lib.audio_tools import resolve_audio_file_extension

        # A crafted name must not smuggle a path out of the temp directory
        with pytest.raises(HTTPException):
            resolve_audio_file_extension("../../etc/passwd")
        assert resolve_audio_file_extension("../../evil/recording.webm") == ".webm"


class TestTranscribeWholeFile:
    """Single-call path: no slicing, no LLM consolidation."""

    async def test_empty_fence_response_normalized_to_silent_chunk(self, tmp_path):
        """The transcribe model may wrap a no-speech response in an empty
        markdown fence; the result must still be treated as silent."""
        emit = AsyncMock()
        pipeline = _make_pipeline(emit)
        source = tmp_path / "source.mp3"
        source.write_bytes(b"fake")

        mock_response = MagicMock()
        mock_response.text = "```plaintext\n```"
        with patch(
            f"{PIPELINE_MODULE}.transcribe_audio_to_text",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await pipeline.transcribe_whole_file(str(source))

        assert result == {"chunk_index": 0, "text": ""}
        emit.assert_awaited_once_with(
            TranscribeStreamEventType.CHUNK_COMPLETED,
            {"chunk_index": 0, "text": ""},
        )

    async def test_normal_response_is_post_processed_and_emitted(self, tmp_path):
        emit = AsyncMock()
        pipeline = _make_pipeline(emit)
        source = tmp_path / "source.mp3"
        source.write_bytes(b"fake")

        mock_response = MagicMock()
        mock_response.text = "这是简体中文"
        with patch(
            f"{PIPELINE_MODULE}.transcribe_audio_to_text",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await pipeline.transcribe_whole_file(str(source))

        assert result == {"chunk_index": 0, "text": "這是簡體中文"}
        emit.assert_awaited_once_with(
            TranscribeStreamEventType.CHUNK_COMPLETED,
            {"chunk_index": 0, "text": "這是簡體中文"},
        )


class TestPostProcessByMode:
    async def test_empty_consolidated_text_short_circuits(self):
        worker = _make_worker(TranscribeMode.REFINED)
        pipeline = MagicMock(
            refine_transcript=AsyncMock(), check_and_fix_punctuation=AsyncMock()
        )

        result = await worker._post_process_by_mode(pipeline, "   ")

        assert result == ""
        pipeline.refine_transcript.assert_not_awaited()
        pipeline.check_and_fix_punctuation.assert_not_awaited()

    async def test_fast_mode_returns_consolidated_text(self):
        worker = _make_worker(TranscribeMode.FAST)
        pipeline = MagicMock(
            refine_transcript=AsyncMock(), check_and_fix_punctuation=AsyncMock()
        )

        result = await worker._post_process_by_mode(pipeline, "some text")

        assert result == "some text"
        pipeline.refine_transcript.assert_not_awaited()
        pipeline.check_and_fix_punctuation.assert_not_awaited()

    async def test_standard_mode_runs_punctuation_fix(self):
        worker = _make_worker(TranscribeMode.STANDARD)
        pipeline = MagicMock(
            check_and_fix_punctuation=AsyncMock(return_value="fixed text")
        )

        result = await worker._post_process_by_mode(pipeline, "raw text")

        assert result == "fixed text"
        pipeline.check_and_fix_punctuation.assert_awaited_once_with("raw text")

    async def test_refined_mode_runs_refine(self):
        worker = _make_worker(TranscribeMode.REFINED)
        pipeline = MagicMock(refine_transcript=AsyncMock(return_value="refined text"))

        result = await worker._post_process_by_mode(pipeline, "raw text")

        assert result == "refined text"
        pipeline.refine_transcript.assert_awaited_once_with("raw text")


class TestCheckAndFixPunctuation:
    async def test_healthy_transcript_skips_fix(self):
        emit = AsyncMock()
        pipeline = _make_pipeline(emit)

        with patch(
            f"{PIPELINE_MODULE}.fix_punctuation", new_callable=AsyncMock
        ) as mock_fix:
            result = await pipeline.check_and_fix_punctuation("這是健康的句子。")

        assert result == "這是健康的句子。"
        mock_fix.assert_not_awaited()
        emit.assert_not_awaited()

    async def test_unhealthy_transcript_emits_event_and_fixes(self):
        emit = AsyncMock()
        pipeline = _make_pipeline(emit)
        unhealthy = "沒有標點的長句子" * 30

        mock_response = MagicMock()
        mock_response.output_text = "修好了。"
        with patch(
            f"{PIPELINE_MODULE}.fix_punctuation",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await pipeline.check_and_fix_punctuation(unhealthy)

        assert result == "修好了。"
        emit.assert_awaited_once_with(
            TranscribeStreamEventType.PUNC_FIXING,
            {"consolidated_text": unhealthy},
        )


class TestRefineTranscript:
    async def test_emits_refining_event_and_returns_refined(self):
        emit = AsyncMock()
        pipeline = _make_pipeline(emit)

        mock_response = MagicMock()
        mock_response.output_text = "refined"
        with patch(
            f"{PIPELINE_MODULE}.refine_transcript",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await pipeline.refine_transcript("raw")

        assert result == "refined"
        emit.assert_awaited_once_with(
            TranscribeStreamEventType.REFINING,
            {"consolidated_text": "raw"},
        )


class TestWorkerRun:
    """TranscribeWorker.run(): single-call path end to end."""

    async def _run(self, tmp_path, pipeline_mock):
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")
        worker = _make_worker(TranscribeMode.STANDARD)

        with patch(
            "app.workers.transcribe_worker.TranscribePipeline",
            return_value=pipeline_mock,
        ):
            await worker.run(str(audio_file))

        emitted = [
            call.args[1]
            for call in worker.transcribe_stream_service.emit_event.await_args_list
        ]
        return worker, emitted, audio_file

    async def test_happy_path_emits_started_then_finished(self, tmp_path):
        pipeline = MagicMock(
            transcribe_whole_file=AsyncMock(
                return_value=TranscribedResultChunk(chunk_index=0, text="raw text")
            ),
            check_and_fix_punctuation=AsyncMock(return_value="raw text"),
        )

        worker, emitted, _ = await self._run(tmp_path, pipeline)

        assert emitted == [
            TranscribeStreamEventType.TASK_STARTED,
            TranscribeStreamEventType.TASK_FINISHED,
        ]
        started_payload = worker.transcribe_stream_service.emit_event.await_args_list[
            0
        ].args[2]
        assert started_payload == {"total_chunks": 1}
        finished_payload = worker.transcribe_stream_service.emit_event.await_args_list[
            1
        ].args[2]
        assert finished_payload == {"final_result": "raw text"}

    async def test_failure_emits_task_failed_and_cleans_up(self, tmp_path):
        pipeline = MagicMock(
            transcribe_whole_file=AsyncMock(side_effect=Exception("boom"))
        )

        _, emitted, audio_file = await self._run(tmp_path, pipeline)

        assert emitted == [
            TranscribeStreamEventType.TASK_STARTED,
            TranscribeStreamEventType.TASK_FAILED,
        ]
        assert not audio_file.exists()  # source file cleaned up in finally
