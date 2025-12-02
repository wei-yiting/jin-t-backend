"""Unit tests for transcribe endpoint validation logic."""

import os
import pytest
from io import BytesIO
from unittest.mock import Mock, AsyncMock, patch
from fastapi import UploadFile
from fastapi.testclient import TestClient

# Mock environment variable before importing main
os.environ["FREE_TIER_OPENAI_API_KEY"] = "test-free-tier-openai-api-key"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

from app.main import app
from app.config import MAX_FILE_SIZE_MB, FREE_TIER_MAX_AUDIO_DURATION_SECONDS


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
    file_size = int((MAX_FILE_SIZE_MB + 1) * 1024 * 1024)
    audio_data = b"x" * file_size
    return ("large_audio.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def mock_transcription():
    """Mock the transcription pipeline and OpenAI client."""
    with patch("app.main.AsyncOpenAI") as mock_openai, \
         patch("app.main.run_transcription_pipeline") as mock_pipeline:
        
        # Mock AsyncOpenAI client
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Mock transcription result
        mock_pipeline.return_value = "This is a test transcription."
        
        yield {
            "openai": mock_openai,
            "pipeline": mock_pipeline,
            "client": mock_client
        }


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
            }
        )
        
        assert response.status_code == 413
        assert f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB}MB" in response.json()["detail"]
    
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
            }
        )
        
        assert response.status_code == 413
        assert f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB}MB" in response.json()["detail"]


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
            }
        )
        
        assert response.status_code == 403
        assert "When not providing a custom API key, you must consent to data collection" in response.json()["detail"]


class TestAudioDurationValidation:
    """Test audio duration validation for free tier."""
    
    def test_free_tier_audio_duration_exceeds_limit(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that audio > 10 minutes is rejected for free tier."""
        duration_over_limit = FREE_TIER_MAX_AUDIO_DURATION_SECONDS + 1
        
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": str(duration_over_limit),
            }
        )
        
        assert response.status_code == 400
        assert "Audio duration exceeds the maximum limit" in response.json()["detail"]
        assert "10 minutes" in response.json()["detail"]
    
    def test_custom_api_key_no_duration_limit(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that audio > 10 minutes is allowed with custom API key."""
        duration_over_limit = FREE_TIER_MAX_AUDIO_DURATION_SECONDS + 100
        
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "test-device-123",
                "X-Consent-Data-Collection": "false",
                "X-Custom-Openai-Api-Key": "sk-test123",
                "X-Audio-Duration": str(duration_over_limit),
            }
        )
        
        assert response.status_code == 200
        assert "transcript" in response.json()


class TestValidRequests:
    """Test valid request scenarios."""
    
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
            }
        )
        
        assert response.status_code == 200
        assert "transcript" in response.json()
        assert response.json()["transcript"] == "This is a test transcription."
    
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
            }
        )
        
        assert response.status_code == 200
        assert "transcript" in response.json()
        
        # Verify custom API key was used
        mock_transcription["openai"].assert_called_with(api_key="sk-test123")
    
    def test_free_tier_uses_env_api_key(
        self, client, small_audio_file, mock_transcription
    ):
        """Test that free tier uses environment variable API key."""
        response = client.post(
            "/transcribe?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "env-key-test-device",
                "X-Consent-Data-Collection": "true",
                "X-Custom-Openai-Api-Key": "",
                "X-Audio-Duration": "100",
            }
        )
        
        assert response.status_code == 200
        
        # Verify environment API key was used
        mock_transcription["openai"].assert_called_with(api_key="test-free-tier-openai-api-key")
