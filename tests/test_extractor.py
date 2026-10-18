"""Tests for `MiBiciExtractor` using a fake catalog page and a stubbed session.

The tests never call mibici.net. `CATALOG_HTML` stands in for the open-data
page, and `_script_get` makes `session.get` return that HTML or CSV bytes.
"""

from pathlib import Path

import pytest
import requests

from mibici_e21.ingestion.extractor import MiBiciExtractor

BASE_URL = "https://www.mibici.net/site/assets/files/"
CATALOG_HTML = """
<a href="/site/assets/files/1079/datos_abiertos_2014_12.csv">Diciembre</a>
<a href="/site/assets/files/1020/datos_abiertos_2016_01-1.csv">Enero</a>
<a href="/site/assets/files/1118/nomenclatura_2026_09.csv">Estaciones</a>
"""


class _Response:
    """Minimal stand-in for the `requests.Response` methods the extractor uses."""

    def __init__(self, text: str = "", chunks: tuple[bytes, ...] = (), ok: bool = True):
        self.text = text
        self._chunks = chunks
        self.ok = ok

    def raise_for_status(self) -> None:
        """Raise when the scripted response represents an HTTP error."""
        if not self.ok:
            raise requests.HTTPError("catalog request failed")

    def iter_content(self, chunk_size: int = 8192):
        """Yield the scripted CSV body one chunk at a time."""
        yield from self._chunks


@pytest.fixture
def extractor(tmp_path: Path) -> MiBiciExtractor:
    """Extractor that writes downloads into a temporary directory."""
    return MiBiciExtractor(base_url=BASE_URL, temp_dir=tmp_path, timeout=5)


def _script_get(extractor: MiBiciExtractor, responses: list[_Response]) -> list[dict]:
    """Replace `session.get` and record each URL, timeout, and stream flag."""
    calls: list[dict] = []

    def get(url: str, timeout: int | None = None, stream: bool | None = None):
        calls.append({"url": url, "timeout": timeout, "stream": stream})
        return responses.pop(0)

    extractor._session.get = get
    return calls


def test_discover_catalog_maps_filenames_to_absolute_urls(extractor: MiBiciExtractor):
    """Catalog keys are CSV filenames and values are absolute file URLs."""
    calls = _script_get(extractor, [_Response(text=CATALOG_HTML)])

    catalog = extractor.discover_catalog()

    assert calls[0]["url"] == "https://www.mibici.net/es/datos-abiertos/"
    assert calls[0]["timeout"] == 5
    assert catalog == {
        "datos_abiertos_2014_12.csv": (
            "https://www.mibici.net/site/assets/files/1079/datos_abiertos_2014_12.csv"
        ),
        "datos_abiertos_2016_01-1.csv": (
            "https://www.mibici.net/site/assets/files/1020/datos_abiertos_2016_01-1.csv"
        ),
        "nomenclatura_2026_09.csv": (
            "https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"
        ),
    }
    assert extractor.catalog is catalog


def test_discover_catalog_raises_when_the_page_fails(extractor: MiBiciExtractor):
    """A failed catalog response is raised instead of building an empty catalog."""
    _script_get(extractor, [_Response(ok=False)])

    with pytest.raises(requests.HTTPError):
        extractor.discover_catalog()


def test_download_trips_uses_the_suffixed_filename(extractor: MiBiciExtractor, tmp_path: Path):
    """January 2016 is fetched from the `-1` URL and saved without that suffix."""
    calls = _script_get(
        extractor,
        [
            _Response(text=CATALOG_HTML),
            _Response(chunks=(b"viaje_id\n", b"1\n")),
        ],
    )

    path = extractor.download_trips(2016, 1)

    assert path == tmp_path / "datos_abiertos_2016_01.csv"
    assert path.read_bytes() == b"viaje_id\n1\n"
    assert calls[1]["url"].endswith("/1020/datos_abiertos_2016_01-1.csv")
    assert calls[1]["stream"] is True


def test_download_trips_raises_when_the_month_is_missing(extractor: MiBiciExtractor):
    """A year and month absent from the catalog stops the download."""
    _script_get(extractor, [_Response(text=CATALOG_HTML)])

    with pytest.raises(StopIteration):
        extractor.download_trips(1999, 1)


def test_download_stations_saves_the_nomenclature_file(extractor: MiBiciExtractor, tmp_path: Path):
    """The station download saves the remote `nomenclatura_` filename."""
    calls = _script_get(
        extractor,
        [
            _Response(text=CATALOG_HTML),
            _Response(chunks=(b"id,name\n",)),
        ],
    )

    path = extractor.download_stations()

    assert path == tmp_path / "nomenclatura_2026_09.csv"
    assert path.read_bytes() == b"id,name\n"
    assert calls[1]["url"].endswith("/1118/nomenclatura_2026_09.csv")


def test_download_file_raises_when_the_name_is_not_in_the_catalog(extractor: MiBiciExtractor):
    """Downloading a filename that is not in the catalog raises `FileNotFoundError`."""
    extractor.catalog = {"datos_abiertos_2014_12.csv": "https://www.mibici.net/x.csv"}

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        extractor._download_file("missing.csv")
