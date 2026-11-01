"""Tests for the data manifest and local data-directory cleanup."""

from dataclasses import replace
import json

import pandas as pd
import pytest

from mibici_e21 import dataset
from mibici_e21.config import DATA_LAKE_SETTINGS, FEATURE_STORE_SETTINGS
from mibici_e21.dataset import _clean_folder, build_data_manifest


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


def _write_parquet(path, rows: int) -> None:
    """Write a one-column Parquet file with `rows` rows."""
    pd.DataFrame({"value": range(rows)}).to_parquet(path, index=False)


def test_build_data_manifest_is_deterministic_and_describes_each_file(tmp_path):
    """The manifest records the pinned revision and the name, hash, size, and rows of a file."""
    demand = tmp_path / "demand_aggregated_2023.parquet"
    _write_parquet(demand, rows=3)

    first = build_data_manifest(2023, {"gold_demand": demand})
    second = build_data_manifest(2023, {"gold_demand": demand})

    assert first == second
    assert first["dataset_revision"] == DATA_LAKE_SETTINGS.HF_DATASET_REVISION
    entry = first["files"]["gold_demand"]
    assert entry["name"] == demand.name
    assert entry["rows"] == 3
    assert entry["size_bytes"] == demand.stat().st_size
    assert len(entry["sha256"]) == 64


def test_read_data_manifest_rejects_a_changed_gold_file(tmp_path, monkeypatch):
    """A gold demand file edited after the ingestion no longer matches its manifest."""
    monkeypatch.setattr(
        dataset, "FEATURE_STORE_SETTINGS", replace(FEATURE_STORE_SETTINGS, GOLD_DIR=tmp_path)
    )
    demand = tmp_path / "demand_aggregated_2023.parquet"
    _write_parquet(demand, rows=3)
    manifest = build_data_manifest(2023, {"gold_demand": demand})
    dataset.data_manifest_path(2023).write_text(json.dumps(manifest), encoding="utf-8")

    assert dataset.read_data_manifest(2023) == manifest

    _write_parquet(demand, rows=4)
    with pytest.raises(ValueError, match="does not match"):
        dataset.read_data_manifest(2023)


def test_read_data_manifest_requires_the_ingestion(tmp_path, monkeypatch):
    """A year without a manifest asks for the ingest entry point."""
    monkeypatch.setattr(
        dataset, "FEATURE_STORE_SETTINGS", replace(FEATURE_STORE_SETTINGS, GOLD_DIR=tmp_path)
    )

    with pytest.raises(FileNotFoundError, match="-e ingest"):
        dataset.read_data_manifest(2023)
