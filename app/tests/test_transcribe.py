"""Integration tests for the task-based transcribe endpoints."""

from app.models import TranscribeMode


FREE_TIER_HEADERS = {
    "X-Device-Id": "valid-free-tier-device",
    "X-Consent-Data-Collection": "true",
    "X-Custom-Openai-Api-Key": "",
    "X-Audio-Duration": "100",
}


class TestStartTranscribeTask:
    """POST /transcribe-tasks creates a task and schedules the worker."""

    def test_valid_free_tier_request(
        self, client, small_audio_file, mock_task_infra
    ):
        response = client.post(
            "/transcribe-tasks?mode=standard",
            files={"audio_file": small_audio_file},
            headers=FREE_TIER_HEADERS,
        )

        assert response.status_code == 200
        task_id = response.json()["task_id"]
        assert task_id

        # Worker constructed with the free-tier key and scheduled in background
        worker_kwargs = mock_task_infra["worker_cls"].call_args.kwargs
        assert worker_kwargs["openai_api_key"] == "test-free-tier-openai-api-key"
        assert worker_kwargs["task_id"] == task_id
        mock_task_infra["worker"].run.assert_awaited_once()
        mock_task_infra["stream"].init_task.assert_awaited_once_with(task_id)

    def test_valid_custom_api_key_request(
        self, client, small_audio_file, mock_task_infra
    ):
        response = client.post(
            "/transcribe-tasks?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "valid-custom-key-device",
                "X-Consent-Data-Collection": "false",
                "X-Custom-Openai-Api-Key": "sk-test123",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 200
        worker_kwargs = mock_task_infra["worker_cls"].call_args.kwargs
        assert worker_kwargs["openai_api_key"] == "sk-test123"

    def test_transcribe_mode_passed_to_worker(
        self, client, small_audio_file, mock_task_infra
    ):
        for mode in ("fast", "standard", "refined"):
            response = client.post(
                f"/transcribe-tasks?mode={mode}",
                files={"audio_file": small_audio_file},
                headers=FREE_TIER_HEADERS,
            )
            assert response.status_code == 200
            worker_kwargs = mock_task_infra["worker_cls"].call_args.kwargs
            assert worker_kwargs["transcribe_mode"] == TranscribeMode(mode)

    def test_consent_enabled_captures_audio_to_r2(
        self, client, small_audio_file, mock_task_infra
    ):
        response = client.post(
            "/transcribe-tasks?mode=standard",
            files={"audio_file": small_audio_file},
            headers=FREE_TIER_HEADERS,
        )

        assert response.status_code == 200
        mock_task_infra["storage"].generate_r2_object_key.assert_called_once()
        mock_task_infra["storage"].capture_raw_audio.assert_awaited_once()

        # The captured content must be the full upload, not a residual chunk
        capture_kwargs = mock_task_infra["storage"].capture_raw_audio.call_args.kwargs
        assert capture_kwargs["file_content"] == b"fake audio data" * 100

        worker_kwargs = mock_task_infra["worker_cls"].call_args.kwargs
        assert worker_kwargs["r2_object_key"] == "test-r2-key"

    def test_no_consent_skips_r2_capture(
        self, client, small_audio_file, mock_task_infra
    ):
        response = client.post(
            "/transcribe-tasks?mode=standard",
            files={"audio_file": small_audio_file},
            headers={
                "X-Device-Id": "valid-custom-key-device",
                "X-Consent-Data-Collection": "false",
                "X-Custom-Openai-Api-Key": "sk-test123",
                "X-Audio-Duration": "100",
            },
        )

        assert response.status_code == 200
        mock_task_infra["storage"].capture_raw_audio.assert_not_awaited()
        worker_kwargs = mock_task_infra["worker_cls"].call_args.kwargs
        assert worker_kwargs["r2_object_key"] is None


class TestGetTaskProgress:
    """GET /transcribe-tasks/{task_id} long-polls the event stream."""

    def test_returns_parsed_stream_messages(self, client, mock_task_infra):
        mock_task_infra["stream"].read_stream.return_value = [
            (
                "stream-key",
                [
                    (
                        "1-0",
                        {
                            "event_type": "TASK_STARTED",
                            "payload": '{"total_chunks": 3}',
                        },
                    ),
                    (
                        "2-0",
                        {
                            "event_type": "CHUNK_COMPLETED",
                            "payload": '{"chunk_index": 0, "text": "hello"}',
                        },
                    ),
                ],
            )
        ]

        response = client.get("/transcribe-tasks/some-task-id?last_id=0-0")

        assert response.status_code == 200
        body = response.json()
        assert body["last_id"] == "2-0"
        assert body["messages"] == [
            {
                "id": "1-0",
                "type": "TASK_STARTED",
                "payload": {"total_chunks": 3},
            },
            {
                "id": "2-0",
                "type": "CHUNK_COMPLETED",
                "payload": {"chunk_index": 0, "text": "hello"},
            },
        ]

    def test_empty_stream_returns_same_last_id(self, client, mock_task_infra):
        mock_task_infra["stream"].read_stream.return_value = []

        response = client.get("/transcribe-tasks/some-task-id?last_id=5-0")

        assert response.status_code == 200
        body = response.json()
        assert body["messages"] == []
        assert body["last_id"] == "5-0"

    def test_stream_error_returns_500(self, client, mock_task_infra):
        mock_task_infra["stream"].read_stream.side_effect = Exception("redis down")

        response = client.get("/transcribe-tasks/some-task-id")

        assert response.status_code == 500
