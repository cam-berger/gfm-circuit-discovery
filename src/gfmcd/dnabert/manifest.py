"""Sibling provenance records for the DNABERT-2 promoter runners.

Every result under ``results/`` must carry a sibling ``<name>.manifest.json``. This module is the
one call site the ``scripts/dnabert_promoter/`` runners use to satisfy that, so the study cannot
drift back into writing bare summaries.

It wraps :class:`gfmcd.provenance.ExperimentManifest` rather than inventing a second shape: the
fields written here are the ones the repository already validates elsewhere (``command``,
``git_commit``, ``git_dirty``, ``created_at_utc``, hashed ``inputs``/``outputs``, ``config``,
``seeds``). The DNABERT-specific part is only *what counts as provenance for this study* — the model
revision, the probe artifact, the split hashes, and the readout layer, which together are what a
reader needs to know a number came from the endpoint it claims.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from gfmcd.provenance import ExperimentManifest, sha256_file

# The canonical split provenance, written once by build_promoter_cohort.py. Recording its hashes on
# every downstream run is what lets a reader tell that two results share a split without trusting
# the filenames.
SPLIT_MANIFEST = Path("results/dnabert_promoter/shared/split_manifest.json")
DATASET_HASHES = Path("results/dnabert_promoter/shared/dataset_sha256.json")

# The splits actually measured are the *cleaned* ones: clean_promoter_cohort.py removes
# conflicting-label and cross-split duplicate hashes and asserts the remainder are disjoint. Recording
# its report alongside the raw hashes is what tells a reader which cohort a number came from — the raw
# split manifest alone would misdescribe every result in the study.
CLEANING_REPORT = Path("results/dnabert_promoter/shared/clean/cleaning_report.json")


def _json_safe(value: Any) -> Any:
    """Coerce a config value into something ``json.dumps`` accepts.

    Runners pass ``vars(args)`` straight through, and 23 of them declare ``type=Path`` options, so
    without this a manifest write would fail at the very end of a long run — the worst possible place
    to discover a serialization bug.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


def sibling_manifest_path(artifact: str | Path) -> Path:
    """Return the ``<stem>.manifest.json`` path beside *artifact*.

    This is the repository-wide convention (247 artifacts) and matches
    ``gfmcd.splicing.pbc_register_stage.sibling_manifest_path``. Note that the five manifests this
    study shipped by hand use an older ``<stem-minus-suffix>_manifest.json`` spelling; those files
    are immutable and stay as they are, but new runs land on the standard name.
    """
    artifact = Path(artifact)
    return artifact.with_name(artifact.stem + ".manifest.json")


def write_run_manifest(
    primary_output: str | Path,
    *,
    experiment_id: str,
    config: Mapping[str, Any],
    inputs: Iterable[str | Path] = (),
    outputs: Iterable[str | Path] = (),
    seed: int | None = None,
    probe_path: str | Path | None = None,
    model_name: str | None = None,
    model_revision: str | None = None,
    selected_layer: int | None = None,
    status: str = "EXPLORATORY",
    notes: Iterable[str] = (),
) -> Path:
    """Write the sibling manifest for *primary_output* and return its path.

    ``primary_output`` is the artifact the manifest belongs to; it is recorded as an output whether
    or not it also appears in *outputs*. Anything the run read that a reader would need in order to
    reproduce it — cohort CSV, probe, upstream summary — belongs in *inputs*, and is hashed.

    ``status`` follows the repository's vocabulary: leave it ``EXPLORATORY`` for smoke, pilot, and
    development output, and set it deliberately for a run whose artifact controls a claim. A
    manifest never promotes its own result.
    """
    recorded: dict[str, Any] = {key: _json_safe(value) for key, value in config.items()}
    if selected_layer is not None:
        recorded["selected_layer"] = selected_layer
    if model_name is not None:
        recorded["model_name"] = model_name
    if model_revision is not None:
        recorded["model_revision"] = model_revision

    # The probe is the readout definition, so its hash is what distinguishes the canonical Layer-2
    # assay from the superseded Layer-3 replicate. Recording the path alone would not.
    if probe_path is not None:
        probe = Path(probe_path)
        recorded["probe_path"] = str(probe)
        if probe.is_file():
            recorded["probe_sha256"] = sha256_file(probe)

    if DATASET_HASHES.is_file():
        recorded["dataset_sha256_manifest"] = str(DATASET_HASHES)
    # Deliberately not called "cohort": the runners pass `vars(args)` and many of them declare a
    # `--cohort` option, so that key is already taken by a path and must not be clobbered.
    recorded["split_leakage_control"] = (
        "leakage_controlled" if CLEANING_REPORT.is_file() else "cleaning_report_absent"
    )

    manifest = ExperimentManifest.create(
        experiment_id,
        [sys.executable if not sys.argv else sys.argv[0], *sys.argv[1:]],
        config=recorded,
        seeds={} if seed is None else {"global": int(seed)},
        status=status,
        notes=list(notes),
    )

    for path in (SPLIT_MANIFEST, DATASET_HASHES, CLEANING_REPORT, *inputs):
        if path is not None and Path(path).is_file():
            manifest.add_input(path)

    seen: set[str] = set()
    for path in (primary_output, *outputs):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        manifest.add_output(path)

    return manifest.write(sibling_manifest_path(primary_output))
