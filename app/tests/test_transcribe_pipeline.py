"""Unit tests for the chunked transcribe pipeline and worker."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.config import (
    INITIAL_CONCURRENT_CHUNK_DURATIONS_SECONDS,
    AUDIO_CHUNK_OVERLAP_MS,
)
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


class TestGetAudioDuration:
    """run_ffmpeg_to_get_audio_duration falls back to decoding when the
    container has no duration header (e.g. browser MediaRecorder uploads)."""

    @staticmethod
    def _proc(returncode: int, stdout: str) -> MagicMock:
        return MagicMock(returncode=returncode, stdout=stdout, stderr="")

    @staticmethod
    def _patch_subprocess(**kwargs):
        # Replace the `subprocess` name inside audio_tools only — patching the
        # global subprocess.run would also intercept unrelated callers
        # (e.g. langsmith collecting git metadata on trace upload).
        from app.lib import audio_tools

        fake = MagicMock()
        fake.PIPE = object()
        fake.run = MagicMock(**kwargs)
        return patch.object(audio_tools, "subprocess", fake), fake

    def test_container_duration_used_when_present(self):
        from app.lib import audio_tools

        patcher, fake = self._patch_subprocess(
            return_value=self._proc(0, '{"format": {"duration": "863.928934"}}')
        )
        with patcher:
            assert audio_tools.run_ffmpeg_to_get_audio_duration("x.mp3") == 863928
        assert fake.run.call_count == 1

    def test_empty_probe_falls_back_to_decoding(self):
        from app.lib import audio_tools

        patcher, fake = self._patch_subprocess(
            side_effect=[
                self._proc(0, '{\n "format": {\n\n }\n}'),
                self._proc(0, "out_time_ms=29994000\nprogress=end\n"),
            ]
        )
        with patcher:
            assert audio_tools.run_ffmpeg_to_get_audio_duration("x.mp3") == 29994
        assert fake.run.call_count == 2

    def test_raises_when_decoding_also_fails(self):
        from app.lib import audio_tools

        patcher, _ = self._patch_subprocess(
            side_effect=[self._proc(0, '{"format": {}}'), self._proc(1, "")]
        )
        with patcher:
            with pytest.raises(Exception, match="by decoding"):
                audio_tools.run_ffmpeg_to_get_audio_duration("x.mp3")


class TestGenerateAudioChunkMetadata:
    def test_zero_duration_returns_no_chunks(self):
        assert TranscribeWorker._generate_audio_chunk_metadata(0) == []

    def test_short_audio_fits_in_single_chunk(self):
        chunks = TranscribeWorker._generate_audio_chunk_metadata(60 * 1000)
        assert chunks == [{"index": 0, "start_ms": 0, "end_ms": 60 * 1000}]

    def test_progressive_sizing_and_overlap(self):
        total_ms = 600 * 1000  # 10 minutes
        chunks = TranscribeWorker._generate_audio_chunk_metadata(total_ms)

        first_duration_ms = INITIAL_CONCURRENT_CHUNK_DURATIONS_SECONDS[0] * 1000
        assert chunks[0]["start_ms"] == 0
        assert chunks[0]["end_ms"] == first_duration_ms

        # Consecutive chunks overlap so the consolidation pass can stitch
        # boundary words that were cut mid-sentence
        for prev, cur in zip(chunks, chunks[1:]):
            assert cur["start_ms"] == prev["end_ms"] - AUDIO_CHUNK_OVERLAP_MS

        assert chunks[-1]["end_ms"] == total_ms
        assert [c["index"] for c in chunks] == list(range(len(chunks)))

    def test_no_zero_length_tail_chunk(self):
        # Exactly the first chunk duration: one chunk, no empty tail
        first_duration_ms = INITIAL_CONCURRENT_CHUNK_DURATIONS_SECONDS[0] * 1000
        chunks = TranscribeWorker._generate_audio_chunk_metadata(first_duration_ms)
        assert len(chunks) == 1
        assert chunks[0]["end_ms"] == first_duration_ms


class TestSliceChunkAndTranscribe:
    async def test_empty_fence_response_normalized_to_silent_chunk(self, tmp_path):
        """The transcribe model may wrap a no-speech response in an empty
        markdown fence; the chunk must still be treated as silent."""
        emit = AsyncMock()
        pipeline = _make_pipeline(emit)
        source = tmp_path / "source.mp3"
        source.write_bytes(b"fake")

        def fake_slice(src, start_ms, end_ms, dest):
            with open(dest, "wb") as f:
                f.write(b"fake-chunk")

        mock_response = MagicMock()
        mock_response.text = "```plaintext\n```"
        with (
            patch(
                f"{PIPELINE_MODULE}.run_ffmpeg_to_slice_audio_file",
                side_effect=fake_slice,
            ),
            patch(
                f"{PIPELINE_MODULE}.transcribe_audio_to_text",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            result = await pipeline.slice_chunk_and_transcribe(
                {"index": 0, "start_ms": 0, "end_ms": 1000}, str(source)
            )

        assert result == {"chunk_index": 0, "text": ""}
        emit.assert_awaited_once_with(
            TranscribeStreamEventType.CHUNK_COMPLETED,
            {"chunk_index": 0, "text": ""},
        )


class TestConsolidateChunksText:
    async def test_all_silent_chunks_skip_llm_call(self):
        pipeline = _make_pipeline()
        silent_chunks = [
            TranscribedResultChunk(chunk_index=0, text=""),
            TranscribedResultChunk(chunk_index=1, text="  "),
        ]

        with patch(
            f"{PIPELINE_MODULE}.consolidate_chunks_text", new_callable=AsyncMock
        ) as mock_consolidate:
            result = await pipeline.consolidate_chunks_text(silent_chunks)

        assert result == ""
        mock_consolidate.assert_not_called()

    async def test_chunks_formatted_in_index_order(self):
        pipeline = _make_pipeline()
        chunks = [
            TranscribedResultChunk(chunk_index=1, text="world"),
            TranscribedResultChunk(chunk_index=0, text="hello"),
        ]

        mock_response = MagicMock()
        mock_response.output_text = "hello world"
        with patch(
            f"{PIPELINE_MODULE}.consolidate_chunks_text",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_consolidate:
            result = await pipeline.consolidate_chunks_text(chunks)

        assert result == "hello world"
        sent_input = mock_consolidate.call_args.args[0]
        assert sent_input == (
            '<chunk index="0">hello</chunk>\n<chunk index="1">world</chunk>'
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


class TestWorkerRunFailure:
    async def test_failure_emits_task_failed_and_cleans_up(self, tmp_path):
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")
        worker = _make_worker(TranscribeMode.STANDARD)

        with patch(
            "app.workers.transcribe_worker.run_ffmpeg_to_get_audio_duration",
            side_effect=Exception("ffprobe exploded"),
        ):
            await worker.run(str(audio_file))

        emitted = worker.transcribe_stream_service.emit_event.await_args_list
        event_types = [call.args[1] for call in emitted]
        assert TranscribeStreamEventType.TASK_FAILED in event_types
        assert not audio_file.exists()  # source file cleaned up in finally
