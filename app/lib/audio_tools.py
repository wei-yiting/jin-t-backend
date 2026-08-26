from pathlib import PurePosixPath

from fastapi import HTTPException

from app.config import SUPPORTED_AUDIO_EXTENSIONS


def resolve_audio_file_extension(filename: str | None) -> str:
    """Return the uploaded file's extension, validated against what the
    transcription API accepts.

    The stored file must keep this extension: the transcription API infers the
    container format from the filename, so writing a `.webm` upload to a
    `.mp3` path makes it reject the audio as corrupted.
    """
    # PurePosixPath keeps a client-supplied name from escaping the temp dir
    extension = PurePosixPath(filename or "").suffix.lower()

    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="不支援的音檔格式，請使用 mp3、wav、m4a、webm 等常見格式",
        )

    return extension


def parse_audio_duration(audio_duration: str) -> float:
    try:
        return float(audio_duration)
    except ValueError:
        return 0.0
    except TypeError:
        return 0.0
