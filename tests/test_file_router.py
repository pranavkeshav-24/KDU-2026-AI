import pytest

from src.utils.file_utils import file_type_for_name, validate_upload


def test_file_type_routes_supported_extensions():
    assert file_type_for_name("report.pdf") == "pdf"
    assert file_type_for_name("chart.png") == "image"
    assert file_type_for_name("meeting.wav") == "audio"


def test_validate_rejects_unsupported_extension():
    valid, message = validate_upload("notes.txt", 10, 100)
    assert not valid
    assert "Supported" in message


def test_file_type_raises_for_unknown_extension():
    with pytest.raises(ValueError):
        file_type_for_name("notes.txt")

