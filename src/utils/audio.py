from datetime import datetime, timezone
from pathlib import Path


def parse_audio_start_datetime(filename: str | Path) -> datetime:
    """Parse the UTC-naive start timestamp embedded in a BIODCASE WAV filename."""
    stem = Path(filename).stem
    if stem.endswith("_000"):
        stem = stem[:-4]
    return datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S")


def parse_annotation_datetime(value: str) -> datetime:
    """Parse annotation datetimes and normalize them to UTC-naive datetimes."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def annotation_offsets_seconds(
    audio_filename: str | Path,
    start_datetime: str,
    end_datetime: str,
) -> tuple[float, float, float]:
    """Return start, end, and duration in seconds relative to the audio file start."""
    audio_start = parse_audio_start_datetime(audio_filename)
    event_start = parse_annotation_datetime(start_datetime)
    event_end = parse_annotation_datetime(end_datetime)
    start_seconds = (event_start - audio_start).total_seconds()
    end_seconds = (event_end - audio_start).total_seconds()
    return start_seconds, end_seconds, end_seconds - start_seconds
