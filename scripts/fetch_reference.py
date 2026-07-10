"""Fetch and verify the reference genome + annotation (TO_FIX.md §12).

Reads data/references/manifest.json, downloads each file to a temp path, verifies (or, on first
run, computes and prints) its SHA-256, decompresses where the install name differs from the
download name, and atomically moves it into place. Refuses to continue on a checksum mismatch.

    uv run python scripts/fetch_reference.py               # download + verify + install
    uv run python scripts/fetch_reference.py --verify-only # check installed files, no download

Checksums are not fabricated: if manifest sha256 is null, the script computes it after a verified
download and prints the value to pin into the manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

from gfmcd.paths import project_paths


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:  # noqa: S310 (trusted refs)
        shutil.copyfileobj(resp, out)


def _decompress_gzip(src: Path, dest: Path) -> None:
    with gzip.open(src, "rb") as fh, dest.open("wb") as out:
        shutil.copyfileobj(fh, out)


def process(name: str, spec: dict, ref_dir: Path, verify_only: bool) -> bool:
    install = ref_dir / spec["install_name"]
    expected = spec.get("sha256")

    if verify_only:
        if not install.exists():
            print(f"  MISSING {install}")
            return False
        if expected is None:
            print(f"  {install.name}: present, no pinned sha256 (computed {sha256_of(install)[:16]}…)")
            return True
        ok = sha256_of(install) == expected
        print(f"  {install.name}: {'OK' if ok else 'MISMATCH'}")
        return ok

    if install.exists() and expected is not None and sha256_of(install) == expected:
        print(f"  {install.name}: already installed and verified")
        return True

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dl = tmp / spec["download_name"]
        _download(spec["url"], dl)

        # The pinned checksum is on the INSTALLED file. Decompress first when needed, then hash.
        staged = tmp / spec["install_name"]
        if spec.get("compression") == "gzip" and spec["download_name"].endswith(".gz") and not spec["install_name"].endswith(".gz"):
            _decompress_gzip(dl, staged)
        else:
            staged = dl if dl.name == spec["install_name"] else staged
            if staged is not dl:
                shutil.copyfile(dl, staged)

        digest = sha256_of(staged)
        if expected is None:
            print(f"  {install.name}: no pinned sha256. Computed: {digest}")
            print(f"    -> PIN THIS into data/references/manifest.json files.{name}.sha256")
        elif digest != expected:
            print(f"  {install.name}: CHECKSUM MISMATCH (got {digest}, expected {expected}) — refusing to install")
            return False

        install.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(install))
        print(f"  installed {install}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true", help="check installed files, do not download")
    args = ap.parse_args()

    ref_dir = project_paths().references
    manifest = json.loads((ref_dir / "manifest.json").read_text())
    print(f"reference dir: {ref_dir}  (override with GFMCD_REFERENCE_DIR)")

    all_ok = True
    for name, spec in manifest["files"].items():
        all_ok &= process(name, spec, ref_dir, args.verify_only)

    print("ALL OK" if all_ok else "INCOMPLETE — see messages above")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
