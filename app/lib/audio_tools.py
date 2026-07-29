import subprocess
import json
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


def parse_audio_duration(audio_duration: str) -> float:
    try:
        return float(audio_duration)
    except ValueError:
        return 0.0
    except TypeError:
        return 0.0

@traceable(run_type="tool", name="Get_Audio_Duration")
def run_ffmpeg_to_get_audio_duration(audio_file_path: str) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration",
        audio_file_path,
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    try:
        if result.returncode != 0:
            raise Exception(
                f"ffprobe failed (returncode={result.returncode}). stderr={result.stderr.strip()!r}"
            )

        data = json.loads(result.stdout)
        duration_seconds = float(data["format"]["duration"])
        return int(duration_seconds * 1000)  # convert to milliseconds
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        stdout_preview = (result.stdout or "").strip().replace("\n", "\\n")[:500]
        stderr_preview = (result.stderr or "").strip().replace("\n", "\\n")[:500]
        raise Exception(
            "Failed to get audio duration: "
            f"{e}. command={command!r}, returncode={result.returncode}, "
            f"stdout={stdout_preview!r}, stderr={stderr_preview!r}"
        )


@traceable(run_type="tool", name="Slice_Audio_File")
def run_ffmpeg_to_slice_audio_file(
    input_path: str,
    start_ms: int,
    end_ms: int,
    output_path: str,
):
    start_seconds = start_ms / 1000
    duration_seconds = (end_ms - start_ms) / 1000

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        input_path,
        "-ss",
        str(start_seconds),
        "-t",
        str(duration_seconds),
        "-acodec",
        "libmp3lame",
        "-q:a",
        "4",
        output_path,
    ]
    subprocess.run(command, check=True)
