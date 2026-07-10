"""Central, machine-independent project paths (TO_FIX.md §11).

Every script and library module should resolve files through :func:`project_paths`
instead of hard-coding ``Path("checkpoints")``, ``"results/reading-frame/..."``, or a
personal ``~/Research/...`` location. Paths resolve from this file's location by default
and are overridable by environment variable, so a clean checkout works from any CWD and on
any host.

Environment overrides:
    GFMCD_ROOT           repository root (default: two parents up from this file)
    GFMCD_DATA_DIR       data/ root
    GFMCD_REFERENCE_DIR  reference genomes (hg38.fa, GENCODE gtf)
    GFMCD_CACHE_DIR      window/codon caches (.npz)
    GFMCD_CHECKPOINT_DIR trained checkpoints (.pt)
    GFMCD_RESULTS_DIR    tracked result dumps + figures
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    references: Path
    cache: Path
    checkpoints: Path
    results: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(os.environ.get("GFMCD_ROOT", Path(__file__).resolve().parents[2]))
        data = Path(os.environ.get("GFMCD_DATA_DIR", root / "data"))
        return cls(
            root=root,
            data=data,
            references=Path(os.environ.get("GFMCD_REFERENCE_DIR", data / "references")),
            cache=Path(os.environ.get("GFMCD_CACHE_DIR", data / "cache")),
            checkpoints=Path(os.environ.get("GFMCD_CHECKPOINT_DIR", root / "checkpoints")),
            results=Path(os.environ.get("GFMCD_RESULTS_DIR", root / "results")),
        )

    # Convenience for the two reference files every genome script needs.
    def hg38(self) -> Path:
        return self.references / "hg38.fa"

    def gencode_gtf(self) -> Path:
        return self.references / "gencode.v50.basic.annotation.gtf.gz"


@lru_cache(maxsize=1)
def project_paths() -> ProjectPaths:
    """The resolved paths for this checkout (cached)."""
    return ProjectPaths.discover()
