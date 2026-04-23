from src.utils.audio import annotation_offsets_seconds, parse_audio_start_datetime


def test_parse_audio_start_datetime():
    parsed = parse_audio_start_datetime("2013-12-25T06-00-00_000.wav")
    assert parsed.year == 2013
    assert parsed.month == 12
    assert parsed.day == 25
    assert parsed.hour == 6


def test_annotation_offsets_seconds():
    start_s, end_s, duration_s = annotation_offsets_seconds(
        "2013-12-25T06-00-00_000.wav",
        "2013-12-25T06:12:04.000000+00:00",
        "2013-12-25T06:12:10.500000+00:00",
    )
    assert start_s == 724.0
    assert end_s == 730.5
    assert duration_s == 6.5

