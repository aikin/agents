import os
import tempfile

from utils.file_loader import FileLoader


def test_load_text():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Test content")
        temp_path = f.name

    try:
        loader = FileLoader()
        content = loader.load_text(temp_path)
        assert content == "Test content"
    finally:
        os.remove(temp_path)


def test_load_text_error():
    loader = FileLoader()
    content = loader.load_text("nonexistent.txt")
    assert "Error" in content
