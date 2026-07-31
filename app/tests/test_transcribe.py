"""Integration tests for transcribe endpoint."""

import os
import pytest
from io import BytesIO
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from openai import BadRequestError, AuthenticationError

# Mock environment variable before importing main
os.environ["FREE_TIER_OPENAI_API_KEY"] = "test-free-tier-openai-api-key"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

from app.main import app
from app.models import TranscribeMode


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def small_audio_file():
    """Create a small valid audio file for testing."""
    # Create a file > 100 bytes but < 25MB
    audio_data = b"fake audio data" * 100  # ~1.5KB
    return ("test_audio.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def mock_transcription():
    """Mock the transcription pipeline and OpenAI client."""
    with (
        patch("app.routers.transcribe.AsyncOpenAI") as mock_openai,
        patch("app.routers.transcribe.run_transcribe_pipeline") as mock_pipeline,
    ):
        # Mock AsyncOpenAI client
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        # Mock transcription result
        mock_pipeline.return_value = "This is a test transcription."

        yield {"openai": mock_openai, "pipeline": mock_pipeline, "client": mock_client}


class TestSuccessScenarios:
    """Test successful request scenarios."""

    def test_valid_free_tier_request(
        self, client, small_audio_file, mock_transcription
    ):
        """Test valid free tier request succeeds."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "valid-free-tier-device",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()
        assert response.json()["transcript"] == "This is a test transcription."

        # Verify environment API key was used
        mock_transcription["openai"].assert_called_with(
            api_key="test-free-tier-openai-api-key"
        )

    def test_valid_custom_api_key_request(
        self, client, small_audio_file, mock_transcription
    ):
        """Test valid custom API key request succeeds."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "valid-custom-key-device",
                "X-Consent-Data-Collection": "false",
                "X-Custom-Openai-Api-Key": "sk-test123",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()

        # Verify custom API key was used
        mock_transcription["openai"].assert_called_with(api_key="sk-test123")

    def test_transcribe_mode_fast(self, client, small_audio_file, mock_transcription):
        """Test transcribe mode FAST."""
        response = client.post(
            "/transcribe?mode=fast",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()

        # Verify pipeline was called with correct mode
        mock_transcription["pipeline"].assert_called_once()
        call_args = mock_transcription["pipeline"].call_args
        assert call_args.kwargs["transcribe_mode"] == TranscribeMode.FAST

    def test_transcribe_mode_standard(
        self, client, small_audio_file, mock_transcription
    ):
        """Test transcribe mode STANDARD."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()

        # Verify pipeline was called with correct mode
        mock_transcription["pipeline"].assert_called_once()
        call_args = mock_transcription["pipeline"].call_args
        assert call_args.kwargs["transcribe_mode"] == TranscribeMode.STANDARD

    def test_transcribe_mode_refined(
        self, client, small_audio_file, mock_transcription
    ):
        """Test transcribe mode REFINED."""
        response = client.post(
            "/transcribe?mode=refined",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()

        # Verify pipeline was called with correct mode
        mock_transcription["pipeline"].assert_called_once()
        call_args = mock_transcription["pipeline"].call_args
        assert call_args.kwargs["transcribe_mode"] == TranscribeMode.REFINED

    def test_pipeline_receives_correct_parameters(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that pipeline receives correct parameters."""
        audio_duration = "150.5"
        transcribe_mode = TranscribeMode.STANDARD

        response = client.post(
            f"/transcribe?mode={transcribe_mode.value}",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": audio_duration,
            },
        )

        assert response.status_code == 200

        # Verify pipeline was called with correct parameters
        mock_transcription["pipeline"].assert_called_once()
        call_args = mock_transcription["pipeline"].call_args

        assert call_args.kwargs["transcribe_mode"] == transcribe_mode
        assert call_args.kwargs["audio_duration"] == audio_duration
        assert "audio_file" in call_args.kwargs
        assert "llm_client" in call_args.kwargs


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_bad_request_error_corrupted_file(
        self, client, small_audio_file, mock_transcription
    ):
        """Test BadRequestError with corrupted file message."""
        # Mock BadRequestError with corrupted message
        error = BadRequestError(
            message="The audio file is corrupted or in an unsupported format",
            response=MagicMock(),
            body=None,
        )
        mock_transcription["pipeline"].side_effect = error

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 400
        assert "corrupted or in an unsupported format" in response.json()["detail"]

    def test_bad_request_error_unsupported_format(
        self, client, small_audio_file, mock_transcription
    ):
        """Test BadRequestError with unsupported format message."""
        # Mock BadRequestError with unsupported message
        error = BadRequestError(
            message="The audio format is unsupported",
            response=MagicMock(),
            body=None,
        )
        mock_transcription["pipeline"].side_effect = error

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 400
        assert "corrupted or in an unsupported format" in response.json()["detail"]

    def test_bad_request_error_other_message(
        self, client, small_audio_file, mock_transcription
    ):
        """Test BadRequestError with other error message."""
        # Mock BadRequestError with other message
        error = BadRequestError(
            message="Some other error occurred",
            response=MagicMock(),
            body=None,
        )
        mock_transcription["pipeline"].side_effect = error

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 400
        assert "Some other error occurred" in response.json()["detail"]

    def test_authentication_error(self, client, small_audio_file, mock_transcription):
        """Test AuthenticationError handling."""
        # Mock AuthenticationError
        error = AuthenticationError(
            message="Invalid API key",
            response=MagicMock(),
            body=None,
        )
        mock_transcription["pipeline"].side_effect = error

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "false",
                "X-Custom-Openai-Api-Key": "sk-invalid-key",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 401
        assert "Invalid OpenAI API key" in response.json()["detail"]

    def test_unexpected_exception(self, client, small_audio_file, mock_transcription):
        """Test unexpected exception handling."""
        # Mock unexpected exception
        mock_transcription["pipeline"].side_effect = Exception("Unexpected error")

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 500
        assert (
            "An unexpected error occurred during transcribe pipeline"
            in response.json()["detail"]
        )
