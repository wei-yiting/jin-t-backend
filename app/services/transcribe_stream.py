import json
from typing import cast
from redis.asyncio import Redis
from redis.exceptions import RedisError
from redis.typing import EncodableT, FieldT
from app.config import TRANSCRIBE_STREAM_EVENT_TTL
from app.models import TranscribeStreamEventType


# Redis `XADD` fields are a dict of field/value pairs. We only emit string keys/values.
# Using a concrete dict type here avoids Pyright invariance issues with redis-py TypeVars.
EventData = dict[str, str | int | float]
RedisStreamEntry = tuple[str, EventData]  # (message_id, fields)
RedisStreamReadResult = list[tuple[str, list[RedisStreamEntry]]]  # [(stream_key, entries)]


class TranscribeStreamService:
    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client


    def _get_key(self, task_id: str) -> str:
        return f"transcribe:stream:{task_id}"


    async def init_task(self, task_id: str) -> None:
        await self.emit_event(task_id, TranscribeStreamEventType.TASK_QUEUED)


    async def emit_event(
        self, task_id: str, event_type: TranscribeStreamEventType, payload: dict | None = None
    ) -> None:
        event_data: EventData = {
            "event_type": event_type.value,
            "payload": json.dumps(payload, ensure_ascii=False),
        }

        stream_key = self._get_key(task_id)
        await self.redis_client.xadd(
            stream_key, cast(dict[FieldT, EncodableT], event_data), maxlen=1000
        )
        await self.redis_client.expire(stream_key, TRANSCRIBE_STREAM_EVENT_TTL)


    async def read_stream(
        self, task_id: str, last_id: str, block_ms: int = 3000
    ) -> RedisStreamReadResult:
        stream_key = self._get_key(task_id)

        try:
            streams = await self.redis_client.xread(
                {stream_key: last_id}, count=100, block=block_ms
            )
            return streams
        except Exception as e:
            raise RedisError(f"Failed to read redis stream: {e}")
