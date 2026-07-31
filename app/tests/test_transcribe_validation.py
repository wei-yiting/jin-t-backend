"""Unit tests for transcribe endpoint validation logic."""

import os
import pytest
from io import BytesIO
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# Mock environment variable before importing main
os.environ["FREE_TIER_OPENAI_API_KEY"] = "test-free-tier-openai-api-key"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

from app.main import app
from app.config import MAX_AUDIO_FILE_SIZE_MB, FREE_TIER_SINGLE_AUDIO_MAX_DURATION_SECONDS


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
def large_audio_file():
    """Create a large audio file exceeding 25MB limit."""
    # Create a file > 25MB
    file_size = int((MAX_AUDIO_FILE_SIZE_MB + 1) * 1024 * 1024)
    audio_data = b"x" * file_size
    return ("large_audio.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def tiny_audio_file():
    """Create a tiny audio file < 100 bytes."""
    audio_data = b"x" * 50  # 50 bytes
    return ("tiny_audio.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def exactly_100_bytes_file():
    """Create an audio file exactly 100 bytes."""
    audio_data = b"x" * 100
    return ("exact_100_bytes.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def exactly_25mb_file():
    """Create an audio file exactly 25MB."""
    file_size = int(MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024)
    audio_data = b"x" * file_size
    return ("exact_25mb.mp3", BytesIO(audio_data), "audio/mpeg")


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


class TestAudioFileValidation:
    """Test audio file basic validation (filename and minimum size)."""

    def test_no_filename_fails(self, client, mock_transcription):
        """Test that request fails when no filename is provided."""
        # FastAPI's File() validation rejects empty filename at parsing stage (422)
        # This happens before our validation logic can run, which is expected behavior
        audio_data = b"fake audio data" * 100
        file_without_name = ("", BytesIO(audio_data), "audio/mpeg")

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": file_without_name},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        # FastAPI returns 422 for invalid file upload format
        assert response.status_code == 422

    def test_file_too_small_fails(self, client, tiny_audio_file):
        """Test that file < 100 bytes is rejected."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": tiny_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 400
        assert "Audio file is too small or corrupted" in response.json()["detail"]

    def test_file_exactly_100_bytes_passes(
        self, client, exactly_100_bytes_file, mock_transcription
    ):
        """Test that file exactly 100 bytes passes validation."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": exactly_100_bytes_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()


class TestFileSizeValidation:
    """Test file size validation for all users."""

    def test_file_size_exceeds_limit_with_custom_api_key(
        self, client, large_audio_file, mock_transcription
    ):
        """Test that file > 25MB is rejected even with custom API key."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": large_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "false",
                "X-Custom-Openai-Api-Key": "sk-test123",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 413
        assert (
            f"File size exceeds the maximum limit of {MAX_AUDIO_FILE_SIZE_MB}MB"
            in response.json()["detail"]
        )

    def test_file_size_exceeds_limit_with_free_tier(
        self, client, large_audio_file, mock_transcription
    ):
        """Test that file > 25MB is rejected for free tier users."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": large_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 413
        assert (
            f"File size exceeds the maximum limit of {MAX_AUDIO_FILE_SIZE_MB}MB"
            in response.json()["detail"]
        )

    def test_file_exactly_25mb_passes(
        self, client, exactly_25mb_file, mock_transcription
    ):
        """Test that file exactly 25MB passes validation."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": exactly_25mb_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()

    def test_file_smaller_than_25mb_passes(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that file < 25MB passes validation."""
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


class TestConsentAndApiKeyValidation:
    """Test consent and API key validation logic."""

    def test_no_consent_no_api_key_fails(self, client, small_audio_file):
        """Test that request fails when no consent and no API key."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "false",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 403
        assert (
            "When not providing a custom API key, you must consent to data collection"
            in response.json()["detail"]
        )

    def test_free_tier_missing_env_api_key_fails(self, client, small_audio_file):
        """Test that free tier fails when environment API key is not configured."""
        # Temporarily remove the environment variable
        original_key = os.environ.get("FREE_TIER_OPENAI_API_KEY")
        if "FREE_TIER_OPENAI_API_KEY" in os.environ:
            del os.environ["FREE_TIER_OPENAI_API_KEY"]

        try:
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
                "Free tier OpenAI API key is not configured on the server"
                in response.json()["detail"]
            )
        finally:
            # Restore the environment variable
            if original_key:
                os.environ["FREE_TIER_OPENAI_API_KEY"] = original_key
            elif "FREE_TIER_OPENAI_API_KEY" not in os.environ:
                os.environ["FREE_TIER_OPENAI_API_KEY"] = "test-free-tier-openai-api-key"


class TestAudioDurationValidation:
    """Test audio duration validation for free tier."""

    def test_free_tier_audio_duration_exceeds_limit(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that audio > 10 minutes is rejected for free tier."""
        duration_over_limit = FREE_TIER_SINGLE_AUDIO_MAX_DURATION_SECONDS + 1

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": str(duration_over_limit),
            },
        )

        assert response.status_code == 400
        assert "Audio duration exceeds the maximum limit" in response.json()["detail"]
        assert "10 minutes" in response.json()["detail"]

    def test_free_tier_audio_duration_exactly_at_limit_passes(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that audio exactly 30 minutes passes for free tier."""
        duration_at_limit = FREE_TIER_SINGLE_AUDIO_MAX_DURATION_SECONDS

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": str(duration_at_limit),
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()

    def test_free_tier_audio_duration_below_limit_passes(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that audio < 30 minutes passes for free tier."""
        duration_below_limit = FREE_TIER_SINGLE_AUDIO_MAX_DURATION_SECONDS - 100

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": str(duration_below_limit),
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()

    def test_custom_api_key_no_duration_limit(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that audio > 30 minutes is allowed with custom API key."""
        duration_over_limit = FREE_TIER_SINGLE_AUDIO_MAX_DURATION_SECONDS + 100

        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "false",
                "X-Custom-Openai-Api-Key": "sk-test123",
                "X-Audio-Duration": str(duration_over_limit),
            },
        )

        assert response.status_code == 200
        assert "transcript" in response.json()

    def test_invalid_duration_format_continues_without_validation(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that invalid duration format continues without validation."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "invalid-duration",
            },
        )

        # Should pass validation and proceed (will fail at pipeline if needed)
        assert response.status_code == 200
        assert "transcript" in response.json()
