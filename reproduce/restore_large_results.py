"""Restore results stored as gzip files due to GitHub's 100 MiB file limit.

Run from anywhere: python reproduce/restore_large_results.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("large_results_manifest.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


for entry in json.loads(MANIFEST.read_text(encoding="utf-8")):
    archive = ROOT / entry["archive"]
    result = ROOT / entry["path"]
    if digest(archive) != entry["archive_sha256"]:
        raise RuntimeError(f"Archive checksum mismatch: {archive}")
    if not result.exists():
        with gzip.open(archive, "rb") as source, result.open("wb") as target:
            shutil.copyfileobj(source, target, 4 * 1024 * 1024)
    if digest(result) != entry["sha256"]:
        raise RuntimeError(f"Result checksum mismatch: {result}")
    print(f"Verified {entry['path']}")
