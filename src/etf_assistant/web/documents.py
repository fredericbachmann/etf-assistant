"""The source PDFs that citations link to.

Only the documents listed in manifest.csv are served, looked up by file name. Never
serve data/ as a directory: it also holds data/facts, the answer key of the evals.
"""

from pathlib import Path

from etf_assistant.corpus.manifest import load_manifest
from etf_assistant.paths import DATA

ROUTE = "/documents"


class Documents:
    def __init__(self, files: dict[tuple[str, str], Path]):
        self._by_key = files  # (isin, doc_type) -> PDF
        self._by_name = {p.name: p for p in files.values()}

    @classmethod
    def from_manifest(cls) -> "Documents":
        """The manifest's PDFs that are present in data/ (they aren't in git)."""
        files = {key: DATA / row["file_path"] for key, row in load_manifest().items()}
        return cls({key: p for key, p in files.items() if p.is_file()})

    def url_for(self, isin: str, doc_type: str) -> str | None:
        """URL of a fund's document ('kid' or 'factsheet'), or None if it isn't available."""
        p = self._by_key.get((isin, doc_type))
        return f"{ROUTE}/{p.name}" if p else None

    def path_of(self, name: str) -> Path | None:
        """The PDF behind a URL's file name; None for anything that isn't a listed document."""
        return self._by_name.get(name)

    def __len__(self) -> int:
        return len(self._by_key)
