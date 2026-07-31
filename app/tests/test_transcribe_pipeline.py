"""Unit tests for transcribe pipeline silent audio early return."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.models import TranscribeMode

pytestmark = pytest.mark.anyio

PIPELINE_MODULE = "app.services.transcribe_pipeline"


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


@pytest.fixture
def mock_audio_file():
    return MagicMock()


@pytest.fixture
def mock_llm_client():
    return AsyncMock()


@pytest.fixture
def default_kwargs(mock_audio_file, mock_llm_client):
    return {
        "audio_file": mock_audio_file,
        "llm_client": mock_llm_client,
        "audio_duration": "10.0",
        "r2_object_key": None,
        "transcribe_request_metadata": {},
    }


def _patch_transcribe(return_text: str):
    mock_response = MagicMock()
    mock_response.text = return_text
    return patch(
        f"{PIPELINE_MODULE}.transcribe_audio_to_text",
        new_callable=AsyncMock,
        return_value=mock_response,
    )


def _patch_refine():
    return patch(
        f"{PIPELINE_MODULE}.refine_transcript",
        new_callable=AsyncMock,
    )


def _patch_fix_punctuation():
    return patch(
        f"{PIPELINE_MODULE}.fix_punctuation",
        new_callable=AsyncMock,
    )


def _patch_langsmith():
    return patch(f"{PIPELINE_MODULE}.get_current_run_tree", side_effect=Exception)


def _patch_wrap_openai():
    return patch(f"{PIPELINE_MODULE}.wrap_openai", side_effect=lambda c: c)


class TestSilentAudioEarlyReturn:
    async def test_refined_mode_empty_transcript_returns_empty(self, default_kwargs):
        with (
            _patch_transcribe(""),
            _patch_refine() as mock_refine,
            _patch_langsmith(),
            _patch_wrap_openai(),
        ):
            from app.services.transcribe_pipeline import run_transcribe_pipeline

            result = await run_transcribe_pipeline(
                **default_kwargs, transcribe_mode=TranscribeMode.REFINED
            )

            assert result == ""
            mock_refine.assert_not_called()

    async def test_standard_mode_empty_transcript_returns_empty(self, default_kwargs):
        with (
            _patch_transcribe(""),
            _patch_fix_punctuation() as mock_fix_punc,
            _patch_langsmith(),
            _patch_wrap_openai(),
        ):
            from app.services.transcribe_pipeline import run_transcribe_pipeline

            result = await run_transcribe_pipeline(
                **default_kwargs, transcribe_mode=TranscribeMode.STANDARD
            )

            assert result == ""
            mock_fix_punc.assert_not_called()

    async def test_refined_mode_whitespace_only_returns_empty(self, default_kwargs):
        with (
            _patch_transcribe("   \n\t  "),
            _patch_refine() as mock_refine,
            _patch_langsmith(),
            _patch_wrap_openai(),
        ):
            from app.services.transcribe_pipeline import run_transcribe_pipeline

            result = await run_transcribe_pipeline(
                **default_kwargs, transcribe_mode=TranscribeMode.REFINED
            )

            assert result == ""
            mock_refine.assert_not_called()

    async def test_refined_mode_normal_transcript_calls_refine(self, default_kwargs):
        mock_refine_response = MagicMock()
        mock_refine_response.output_text = "Refined text"

        with (
            _patch_transcribe("這是一段正常的語音"),
            _patch_refine() as mock_refine,
            _patch_langsmith(),
            _patch_wrap_openai(),
        ):
            mock_refine.return_value = mock_refine_response

            from app.services.transcribe_pipeline import run_transcribe_pipeline

            result = await run_transcribe_pipeline(
                **default_kwargs, transcribe_mode=TranscribeMode.REFINED
            )

            assert result == "Refined text"
            mock_refine.assert_called_once()

    async def test_fast_mode_empty_transcript_returns_empty(self, default_kwargs):
        with (
            _patch_transcribe(""),
            _patch_langsmith(),
            _patch_wrap_openai(),
        ):
            from app.services.transcribe_pipeline import run_transcribe_pipeline

            result = await run_transcribe_pipeline(
                **default_kwargs, transcribe_mode=TranscribeMode.FAST
            )

            assert result == ""

    async def test_fast_mode_normal_transcript_returns_post_processed(
        self, default_kwargs
    ):
        with (
            _patch_transcribe("这是简体中文"),
            _patch_langsmith(),
            _patch_wrap_openai(),
        ):
            from app.services.transcribe_pipeline import run_transcribe_pipeline

            result = await run_transcribe_pipeline(
                **default_kwargs, transcribe_mode=TranscribeMode.FAST
            )

            assert result != ""
            assert result == "這是簡體中文"
