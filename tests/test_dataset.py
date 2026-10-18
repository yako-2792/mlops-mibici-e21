"""Tests for local data-directory cleanup."""

from mibici_e21.dataset import _clean_folder


def test_clean_folder_removes_contents_and_keeps_the_directory(tmp_path):
    """Files and subdirectories are removed, and the folder and `.gitkeep` remain."""
    (tmp_path / ".gitkeep").write_bytes(b"")
    (tmp_path / "trip.parquet").write_bytes(b"parquet")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "note.txt").write_text("keep the parent", encoding="utf-8")

    _clean_folder(tmp_path)

    assert tmp_path.exists()
    assert [item.name for item in tmp_path.iterdir()] == [".gitkeep"]


def test_clean_folder_logs_when_the_directory_is_missing(tmp_path):
    """A path that does not exist is left untouched."""
    missing = tmp_path / "missing"

    _clean_folder(missing)

    assert not missing.exists()
