import subprocess
import json
from langsmith import traceable


def parse_audio_duration(audio_duration: str) -> float:
    try:
        return float(audio_duration)
    except ValueError:
        return 0.0
    except TypeError:
        return 0.0

# Not @traceable: this runs on the task-start critical path and LangSmith
# posting adds erratic multi-second latency; the duration is already visible
# in the chunk metadata of the traced pipeline spans.
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

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            duration_seconds = float(data["format"]["duration"])
            return int(duration_seconds * 1000)  # convert to milliseconds
        except (KeyError, ValueError, json.JSONDecodeError):
            # Live-recorded uploads (e.g. browser MediaRecorder) stream their
            # container and never write a duration header, so the probe comes
            # back empty; fall through to measuring by decoding.
            pass

    return _measure_audio_duration_by_decoding(audio_file_path)


def _measure_audio_duration_by_decoding(audio_file_path: str) -> int:
    """Decode the audio through the null muxer and read the final progress
    timestamp. Slower than probing the header, but works for containers that
    carry no duration metadata."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostats",
        "-progress",
        "pipe:1",
        "-i",
        audio_file_path,
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    last_out_time_us = None
    for line in result.stdout.splitlines():
        # ffmpeg's out_time_ms is microseconds despite the name
        if line.startswith("out_time_ms="):
            try:
                last_out_time_us = int(line.split("=", 1)[1])
            except ValueError:
                continue

    if result.returncode != 0 or last_out_time_us is None or last_out_time_us <= 0:
        stderr_preview = (result.stderr or "").strip().replace("\n", "\\n")[:500]
        raise Exception(
            "Failed to get audio duration by decoding: "
            f"command={command!r}, returncode={result.returncode}, "
            f"out_time={last_out_time_us!r}, stderr={stderr_preview!r}"
        )

    return last_out_time_us // 1000


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
