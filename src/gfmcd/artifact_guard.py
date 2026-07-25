"""Refuse to overwrite an existing result artifact.

`README.md` states "no new run may silently overwrite the scientific record" and
`REPRODUCE.md` §4 says "Write a new immutable artifact; do not overwrite the controlling
result." Until this module existed, nothing enforced either: **56 runners** default `--out`
to a path listed as an artifact in `results/MANIFEST.csv` (41 of them `canonical`), and
`grep -rc "out).exists()" scripts/` returned zero files. Reproducing a claim by its
documented command destroyed the artifact it was meant to be compared against.

Why this is a separate module rather than part of `provenance.py`
-----------------------------------------------------------------
`src/gfmcd/provenance.py` is **hash-pinned by C36's frozen confirmation package**
(`results/phase-lattice/p3_phase_surgery_confirmation_package_v1.json`). Editing it to add
this guard broke that seal on the first attempt and was caught by
`tests/test_hash_pin_ledger.py` — the guard catching its own author. Rather than widen the
known-broken ledger to accommodate a *new, avoidable* drift, the guard lives here, in an
unpinned module, and `provenance.py` is left byte-identical to the state C36 recorded.

Consequence to be explicit about: `ExperimentManifest.write` is therefore **not** guarded.
That is a smaller loss than it sounds — `write` persists *manifests*, whereas the hazard is
the *result* JSONs, which runners write by their own means and which never passed through
it. The real protection for the existing 56 is the inventory assertion in
`tests/test_overwrite_guard.py`, which fails the day a 57th appears.

Usage in new runners (call before writing anything):

    from gfmcd.artifact_guard import guard_output_path
    out = guard_output_path(args.out, overwrite=args.overwrite_record)

See `CORRECTIONS.md` R35.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ArtifactExistsError", "guard_output_path"]


class ArtifactExistsError(FileExistsError):
    """Refusal to overwrite an existing artifact. Results in this repo are immutable."""


def guard_output_path(path: str | Path, *, overwrite: bool = False) -> Path:
    """Return *path*, refusing if it already exists unless *overwrite* is set.

    Refuses rather than auto-suffixing. An auto-suffixed sibling looks like a successful
    rerun while quietly not being the artifact anyone cites; a refusal makes the operator
    choose a new name, which is the decision the immutability rule is actually asking for.

    Pure predicate: creates nothing, makes no directories.

    Raises:
        ArtifactExistsError: if *path* exists and *overwrite* is False. Subclasses
            ``FileExistsError``, so callers already handling that keep working.
    """
    p = Path(path)
    if p.exists() and not overwrite:
        raise ArtifactExistsError(
            f"refusing to overwrite existing artifact {p}: results are immutable. "
            "Write a new versioned filename, or pass overwrite=True "
            "(runners: --overwrite-record) if you genuinely intend to replace the record."
        )
    return p
