import json
from redis.asyncio import Redis
from redis.exceptions import RedisError
from app.config import TRANSCRIBE_TASK_PROGRESS_TTL
from app.models import TaskStatus, TaskProcessingProgressCode


class TaskProgressService:
    ttl = TRANSCRIBE_TASK_PROGRESS_TTL

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client

    def _get_key(self, task_id: str) -> str:
        return f"task:{task_id}"

    async def init_task(self, task_id: str):
        initial_state = {
            "status": TaskStatus.QUEUED.value,
            "message": "音檔上傳中...",
        }
        await self.redis_client.set(
            self._get_key(task_id), json.dumps(initial_state), ex=self.ttl
        )

    async def update_progress(
        self, task_id: str, progress_code: TaskProcessingProgressCode, message: str
    ):
        state = {
            "status": TaskStatus.PROCESSING.value,
            "progress_code": progress_code.value,
            "message": message,
        }
        await self.redis_client.set(
            self._get_key(task_id), json.dumps(state), ex=self.ttl
        )

    async def mark_completed(self, task_id: str, result_transcript: str):
        state = {
            "status": TaskStatus.COMPLETED,
            "message": "處理完成",
            "transcript": result_transcript,
        }
        await self.redis_client.set(
            self._get_key(task_id), json.dumps(state), ex=self.ttl
        )

    async def mark_failed(self, task_id: str, error_message: str):
        state = {
            "status": TaskStatus.FAILED,
            "message": "轉錄失敗",
            "error_detail": error_message,
        }
        await self.redis_client.set(
            self._get_key(task_id), json.dumps(state), ex=self.ttl
        )

    async def get_task_progress(self, task_id: str) -> dict[str, str] | None:
        try:
            task_data = await self.redis_client.get(self._get_key(task_id))
            if not task_data:
                return None
            return json.loads(task_data)
        except json.JSONDecodeError:
            print(f"Error: Redis data for {task_id} is not valid JSON.")
            return {"status": "failed", "message": "Internal Data Error"}
        except RedisError as e:
            print(f"Error getting task progress: {e}")
            return None

        except Exception as e:
            print(f"Error getting task progress: {e}")
            return None
