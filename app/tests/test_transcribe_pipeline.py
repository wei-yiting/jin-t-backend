"""Unit tests for transcribe pipeline silent audio early return."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import anyio

from app.models import TranscribeMode


PIPELINE_MODULE = "app.services.transcribe_pipeline"


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


class TestSilentAudioEarlyReturn:
    def test_refined_mode_empty_transcript_returns_empty(self, default_kwargs):
        async def _run():
            with (
                _patch_transcribe(""),
                _patch_refine() as mock_refine,
                _patch_langsmith(),
            ):
                from app.services.transcribe_pipeline import run_transcribe_pipeline

                result = await run_transcribe_pipeline(
                    **default_kwargs, transcribe_mode=TranscribeMode.REFINED
                )

                assert result == ""
                mock_refine.assert_not_called()

        anyio.run(_run)

    def test_standard_mode_empty_transcript_returns_empty(self, default_kwargs):
        async def _run():
            with (
                _patch_transcribe(""),
                _patch_fix_punctuation() as mock_fix_punc,
                _patch_langsmith(),
            ):
                from app.services.transcribe_pipeline import run_transcribe_pipeline

                result = await run_transcribe_pipeline(
                    **default_kwargs, transcribe_mode=TranscribeMode.STANDARD
                )

                assert result == ""
                mock_fix_punc.assert_not_called()

        anyio.run(_run)

    def test_refined_mode_whitespace_only_returns_empty(self, default_kwargs):
        async def _run():
            with (
                _patch_transcribe("   \n\t  "),
                _patch_refine() as mock_refine,
                _patch_langsmith(),
            ):
                from app.services.transcribe_pipeline import run_transcribe_pipeline

                result = await run_transcribe_pipeline(
                    **default_kwargs, transcribe_mode=TranscribeMode.REFINED
                )

                assert result == ""
                mock_refine.assert_not_called()

        anyio.run(_run)

    def test_refined_mode_normal_transcript_calls_refine(self, default_kwargs):
        async def _run():
            mock_refine_response = MagicMock()
            mock_refine_response.output_text = "Refined text"

            with (
                _patch_transcribe("這是一段正常的語音"),
                _patch_refine() as mock_refine,
                _patch_langsmith(),
            ):
                mock_refine.return_value = mock_refine_response

                from app.services.transcribe_pipeline import run_transcribe_pipeline

                result = await run_transcribe_pipeline(
                    **default_kwargs, transcribe_mode=TranscribeMode.REFINED
                )

                assert result == "Refined text"
                mock_refine.assert_called_once()

        anyio.run(_run)
