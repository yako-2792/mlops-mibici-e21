"""Download MiBici open-data CSVs listed on the public catalog page."""

from pathlib import Path
import re
from urllib.parse import urljoin

from loguru import logger
import requests

from mibici_e21.config import DATA_LAKE_SETTINGS

_CSV_HREF_PATTERN = re.compile(r'href="([^"]+/([^"]+\.csv))"')


class MiBiciExtractor:
    """Download trip and station CSVs from the MiBici open-data catalog.

    The file id in each URL is read from the catalog HTML. It is not derived
    from the year and month.
    """

    def __init__(
        self,
        base_url: str = DATA_LAKE_SETTINGS.MIBICI_BASE_URL,
        temp_dir: Path = DATA_LAKE_SETTINGS.TEMP_DIR,
        timeout: int = 60,
    ):
        """Create an extractor.

        - `base_url`: MiBici files directory. The catalog page is resolved from its origin.
        - `temp_dir`: directory where downloaded CSVs are written.
        - `timeout`: seconds to wait for each HTTP request.
        """
        self.base_url = base_url
        self.temp_dir = temp_dir
        self.timeout = timeout
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html",
            }
        )

        self.catalog: dict[str, str] = {}

    def download_trips(self, year: int, month: int) -> Path:
        """Download the trip CSV for one month.

        The catalog is matched by the `datos_abiertos_YYYY_MM` prefix. The file
        is saved as `datos_abiertos_YYYY_MM.csv`, whatever suffix the remote name has.
        Raises `FileNotFoundError` when the month is not in the catalog.
        """
        prefix = f"datos_abiertos_{year}_{month:02d}"
        return self._download_file(self._find_in_catalog(prefix), target_filename=f"{prefix}.csv")

    def download_stations(self) -> Path:
        """Download the station catalog CSV whose name starts with `nomenclatura_`."""
        return self._download_file(self._find_in_catalog("nomenclatura_"))

    def discover_catalog(self) -> dict[str, str]:
        """Read the open-data page and map each CSV filename to its absolute URL."""
        catalog_url = urljoin(self.base_url, "/es/datos-abiertos/")
        response = self._session.get(catalog_url, timeout=self.timeout)
        response.raise_for_status()

        matches = _CSV_HREF_PATTERN.findall(response.text)
        self.catalog = {filename: urljoin(self.base_url, href) for href, filename in matches}
        logger.info(f"Catalog loaded: {len(self.catalog)} CSV files")
        return self.catalog

    def _find_in_catalog(self, prefix: str) -> str:
        """Return the first catalog filename that starts with `prefix`.

        Raises `FileNotFoundError` when no filename matches.
        """
        if not self.catalog:
            self.discover_catalog()

        filename = next((name for name in self.catalog if name.startswith(prefix)), None)
        if filename is None:
            raise FileNotFoundError(f"No file starting with {prefix} in the MiBici catalog")
        return filename

    def _download_file(self, file_name: str, target_filename: str | None = None) -> Path:
        """Stream one catalog file into `temp_dir`.

        `file_name` is the remote catalog key. `target_filename` is the local
        name; when omitted, the remote name is kept.
        """
        if not self.catalog:
            self.discover_catalog()

        if file_name not in self.catalog:
            raise FileNotFoundError(f"File {file_name} not found in catalog")

        local_filename = target_filename or file_name
        target_path = self.temp_dir / local_filename

        download_url = self.catalog[file_name]
        logger.info(f"Downloading {local_filename}")
        response = self._session.get(download_url, timeout=self.timeout, stream=True)
        response.raise_for_status()

        with open(target_path, "wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))

        logger.info(f"Downloaded {local_filename} ({_format_size(target_path.stat().st_size)})")
        return target_path


def _format_size(size_bytes: int) -> str:
    """Format a byte count as KB or MB."""
    size_kb = size_bytes / 1024
    if size_kb >= 1024:
        return f"{size_kb / 1024:.1f} MB"
    return f"{size_kb:.0f} KB"
