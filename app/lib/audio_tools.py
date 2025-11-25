from fastapi import UploadFile
from langsmith import traceable

from app.models import AudioMetadataFromRequest


@traceable(run_type="tool", name="Get_Audio_File_Metadata")
def get_audio_metadata(audio_file: UploadFile) -> AudioMetadataFromRequest:
    content_type = audio_file.content_type
    raw_file = audio_file.file

    # Get file size (Bytes)
    raw_file.seek(0, 2)
    file_size_bytes = raw_file.tell()
    raw_file.seek(0)

    return {
        "audio_file_content_type": content_type,
        "audio_file_size_mb": round(file_size_bytes / (1024 * 1024), 3),
    }
