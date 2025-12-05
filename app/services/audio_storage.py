import boto3
import os
import datetime
from mimetypes import guess_type
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


class AudioStorageService:
    def __init__(self):
        self.r2_client = boto3.client(
            service_name="s3",
            endpoint_url=os.getenv("R2_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
        )
        self.bucket_name = os.getenv("R2_BUCKET_NAME")

    def generate_r2_object_key(
        self, device_id: str, request_id: str, filename: str | None
    ) -> str:
        now = datetime.datetime.now(datetime.UTC)
        ext = filename.split(".")[-1] if filename and "." in filename else "wav"
        return f"{now.year}/{now.month:02d}/{now.day:02d}/{now.strftime('%H%M%S')}_{device_id}_{request_id}.{ext}"

    def capture_raw_audio(
        self,
        file_content: bytes,
        r2_object_key: str,
        original_filename: str | None,
        content_type: str | None,
    ) -> str:
        if original_filename and not content_type:
            content_type = guess_type(original_filename)[0]
            content_type = content_type or "application/octet-stream"

        request_id = r2_object_key.split(".")[0].split("_")[-1]

        try:
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=r2_object_key,
                Body=file_content,
                ContentType=content_type,
            )
            print(
                f"Request Id: {request_id} capture raw audio to R2 successful: {r2_object_key}"
            )
            return r2_object_key

        except ClientError as e:
            print(
                f"Request Id: {request_id} capture raw audio to R2 failed (ClientError): {e}"
            )
            return ""

        except Exception as e:
            print(
                f"Request Id: {request_id} capture raw audio to R2 failed (Unknown Error): {e}"
            )
            return ""


audio_storage_service = AudioStorageService()
