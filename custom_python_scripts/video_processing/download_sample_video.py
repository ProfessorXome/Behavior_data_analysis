#!/usr/bin/env python3
"""
Download the raw sample video, 071026_HML1630_01.avi (6.5 GB), from Zenodo.

The video is too large for GitHub, so it is published separately on Zenodo
(record 24546189, https://doi.org/10.5281/zenodo.24546189, CC BY 4.0). It is
the raw recording that Tierpsy Tracker processed into the sample data's
Videos/MaskedVideos/ and Videos/Results/ files. No script in this repository
reads it; download it only if you want to watch it or re-run Tierpsy on it.

By default it is saved to ../Sample_data/Videos/ (next to the Tierpsy output
it came from). The download resumes if interrupted (re-run the same command),
and the file is checked against the MD5 checksum Zenodo lists for it.

Uses only the Python standard library, so any Python 3.8+ works.

Examples (run from custom_python_scripts/)
--------
  python video_processing/download_sample_video.py
  python video_processing/download_sample_video.py --dest /path/to/folder
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]  # custom_python_scripts/
DEFAULT_DEST = SCRIPTS_DIR.parent / "Sample_data" / "Videos"

ZENODO_RECORD = 24546189
FILE_NAME = "071026_HML1630_01.avi"
FILE_SIZE = 6_528_068_212  # bytes
FILE_MD5 = "e26f2559b696dba7c370a3ca2455740b"
URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{FILE_NAME}/content"

CHUNK = 8 * 1024 * 1024  # for checksumming
NET_CHUNK = 1024 * 1024  # for downloading
MAX_ATTEMPTS = 5
USER_AGENT = "worm-analysis-sample-download"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Download {FILE_NAME} from Zenodo record {ZENODO_RECORD}.",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Folder to save the video in (default: {DEFAULT_DEST}).",
    )
    return p.parse_args()


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def download_to(part: Path) -> None:
    """Download URL into `part`, resuming from its current size."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= FILE_SIZE:
            return
        req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:
                    # Server ignored the range request: start over.
                    have = 0
                mode = "ab" if have else "wb"
                if have:
                    print(f"Resuming at {gb(have)}")
                last = 0.0
                with open(part, mode) as out:
                    while True:
                        block = resp.read(NET_CHUNK)
                        if not block:
                            break
                        out.write(block)
                        have += len(block)
                        now = time.monotonic()
                        if now - last >= 2:
                            pct = 100 * have / FILE_SIZE
                            print(f"\r  {gb(have)} / {gb(FILE_SIZE)} ({pct:.1f}%)", end="", flush=True)
                            last = now
                print(f"\r  {gb(have)} / {gb(FILE_SIZE)} (100.0%)" if have >= FILE_SIZE else "")
            if have >= FILE_SIZE:
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                if exc.code == 416:
                    return  # range starts at the end: nothing left to fetch
                if 400 <= exc.code < 500 and exc.code not in (408, 429):
                    raise SystemExit(f"Zenodo refused the download: {exc}") from exc
            print(f"\nDownload interrupted ({exc}).", file=sys.stderr)
        if attempt < MAX_ATTEMPTS:
            print(f"Retrying ({attempt + 1}/{MAX_ATTEMPTS}) in 10 s ...", file=sys.stderr)
            time.sleep(10)
    raise SystemExit(
        f"Download did not finish after {MAX_ATTEMPTS} attempts. "
        "Run the same command again to resume."
    )


def main() -> None:
    args = parse_args()
    dest_dir = args.dest.expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / FILE_NAME
    part = dest_dir / (FILE_NAME + ".part")

    print(f"Zenodo record {ZENODO_RECORD}: {FILE_NAME} ({gb(FILE_SIZE)})")
    print(f"Destination: {final}")

    if final.exists():
        print("Already present; checking MD5 ...")
        if md5sum(final) == FILE_MD5:
            print("OK: file matches the Zenodo checksum.")
            return
        raise SystemExit(
            f"{final} exists but does not match the Zenodo checksum. "
            "Move or delete it, then run this again."
        )

    have = part.stat().st_size if part.exists() else 0
    free = shutil.disk_usage(dest_dir).free
    if FILE_SIZE - have > free:
        raise SystemExit(
            f"Not enough disk space: need {gb(FILE_SIZE - have)}, "
            f"{gb(free)} free in {dest_dir}."
        )

    download_to(part)

    print("Checking MD5 ...")
    size = part.stat().st_size
    if size != FILE_SIZE or md5sum(part) != FILE_MD5:
        part.unlink()
        raise SystemExit(
            f"Downloaded file does not match Zenodo's size/checksum "
            f"({size} bytes); it was deleted. Run this again to retry."
        )
    part.rename(final)
    print(f"OK: saved {final}")


if __name__ == "__main__":
    main()
