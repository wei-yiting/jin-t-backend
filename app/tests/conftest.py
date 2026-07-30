"""Shared fixtures for transcribe endpoint tests."""

import os
import pytest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# Mock environment variables before importing main
os.environ["FREE_TIER_OPENAI_API_KEY"] = "test-free-tier-openai-api-key"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

# Tests must never post traces to LangSmith, even when the developer's shell
# has tracing enabled — @traceable would otherwise do network I/O per call.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from app.main import app
from app.config import MAX_AUDIO_FILE_SIZE_MB


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


class FakeRedisPipeline:
    """Stands in for redis.asyncio pipeline in the rate-limit dependency."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def hgetall(self, key):
        pass

    def ttl(self, key):
        pass

    def hincrby(self, *args):
        pass

    def hincrbyfloat(self, *args):
        pass

    def expire(self, *args):
        pass

    async def execute(self):
        # Alternating (hgetall, ttl) results; empty data → no limits hit.
        return [{}, -2] * 10


class FakeRedis:
    def pipeline(self):
        return FakeRedisPipeline()


@pytest.fixture
def client():
    """Create test client with a fake Redis on app state."""
    app.state.redis = FakeRedis()
    return TestClient(app)


@pytest.fixture
def small_audio_file():
    """Create a small valid audio file for testing."""
    audio_data = b"fake audio data" * 100  # ~1.5KB
    return ("test_audio.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def large_audio_file():
    """Create a large audio file exceeding the size limit."""
    file_size = int((MAX_AUDIO_FILE_SIZE_MB + 1) * 1024 * 1024)
    audio_data = b"x" * file_size
    return ("large_audio.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def tiny_audio_file():
    """Create a tiny audio file < 100 bytes."""
    audio_data = b"x" * 50
    return ("tiny_audio.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def exactly_100_bytes_file():
    """Create an audio file exactly 100 bytes."""
    audio_data = b"x" * 100
    return ("exact_100_bytes.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def exactly_25mb_file():
    """Create an audio file exactly at the size limit."""
    file_size = int(MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024)
    audio_data = b"x" * file_size
    return ("exact_25mb.mp3", BytesIO(audio_data), "audio/mpeg")


@pytest.fixture
def mock_task_infra():
    """Mock the task infrastructure the router hands off to.

    The router only creates the task: it stores the upload, initializes the
    stream, and schedules TranscribeWorker.run as a background task.
    """
    with (
        patch("app.routers.transcribe.TranscribeWorker") as mock_worker_cls,
        patch("app.routers.transcribe.TranscribeStreamService") as mock_stream_cls,
        patch("app.routers.transcribe.audio_storage_service") as mock_storage,
        patch(
            "app.dependencies.core_decode_and_decrypt_openai_api_key",
            side_effect=lambda key: key,
        ),
    ):
        mock_worker = MagicMock()
        mock_worker.run = AsyncMock()
        mock_worker_cls.return_value = mock_worker

        mock_stream = MagicMock()
        mock_stream.init_task = AsyncMock()
        mock_stream.read_stream = AsyncMock(return_value=[])
        mock_stream_cls.return_value = mock_stream

        mock_storage.generate_r2_object_key.return_value = "test-r2-key"
        mock_storage.capture_raw_audio = AsyncMock()

        yield {
            "worker_cls": mock_worker_cls,
            "worker": mock_worker,
            "stream_cls": mock_stream_cls,
            "stream": mock_stream,
            "storage": mock_storage,
        }
