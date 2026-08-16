#!/usr/bin/env python
"""Build the compact R3d train exposure registry without running a model.

The registry has a deliberately narrow, extensional meaning.  It contains exactly two
registered direct-exposure classes: Claude's inverse-acceptor M1/M2/M3 work and all 400 rows
of the clean-alt head screen.  It is *not* an inventory of every train example ever seen by
this repository, and in particular omits HCA18/the broader Atlas, donor-circuit studies, and
earlier R3 train scoring used for candidate generation.

Source A is reconstructed from the genome rather than trusted from prose because several
Claude result files omit row identities.  The four deterministic cohorts are flank k=3,
flank k=1, support-aware k=1, and the later support-aware k=3 M1 cohort.  The current
96-head artifact is checked to use exactly the already-counted flank-k=1 rows.  Source B is
all 400 ``head_screen_clean_alt`` shards because both target strata contributed to reverse-
head candidate selection.  Acceptor and donor rows remain separately audited so exact-
endpoint exposure is not conflated with the broader candidate-selection exposure.

The GTF is restricted to train chromosomes before the FASTA object is asked for sequence.
No model or checkpoint is imported or evaluated.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import sha256_file  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    GAP_MAX,
    GAP_MIN,
    _dataset,
    matched_decoy_donor,
)
from gfmcd.splicing.coupling import (  # noqa: E402
    CouplingConfig,
    MOTIF_CLEAN_DONOR_ALT,
    planted_donor_motifs,
)
from gfmcd.splicing.reverse_coupling import donor_alt_admissible  # noqa: E402


RESULT_SCHEMA = "gfmcd.r3d.exposed_gene_registry/1"
ARTIFACT_REVISION = "r3d-train-direct-exposure-registry-v1"
REGISTRY_SCOPE_ID = (
    "claude_inverse_acceptor_and_clean_alt_head_screen_direct_exposures_v1"
)
MATCHED_OPERATOR = "GG"
DECOY_TOL = 40
ROW_RANK_SALT = "r3d-train-one-row-per-gene-v1"
GENE_RANK_SALT = "r3d-train-gene-order-v1"

CANONICAL_CHROMOSOMES = tuple(
    [f"chr{index}" for index in range(1, 23)] + ["chrX", "chrY"]
)
VALIDATION_CHROMOSOMES = tuple(CouplingConfig().val_chromosomes)
TEST_CHROMOSOMES = tuple(CouplingConfig().test_chromosomes)
FORBIDDEN_CHROMOSOMES = frozenset(
    (*VALIDATION_CHROMOSOMES, *TEST_CHROMOSOMES)
)
TRAIN_CHROMOSOMES = tuple(
    chromosome
    for chromosome in CANONICAL_CHROMOSOMES
    if chromosome not in FORBIDDEN_CHROMOSOMES
)

CLAUDE_RESULT_NAMES = (
    "m1_development.json",
    "m1_development_k1.json",
    "m2m3_development.json",
    "m3b_development.json",
    "m1_v2.json",
    "m2m3_v2.json",
    "m3b_v2.json",
    "m2m3_allheads.json",
)
CLAUDE_MANIFEST_NAMES = tuple(
    name.replace(".json", ".manifest.json") for name in CLAUDE_RESULT_NAMES
)

RECONSTRUCTION_BLOB_SPECS = (
    {
        "label": "original_matcher_7222",
        "revision": "7222b8178363cc40acfaaab371733d23ed818cf4",
        "path": "src/gfmcd/splicing/donor_class.py",
        "size_bytes": 6020,
        "sha256": "91c411f881bfc63af794503a19f05b9d84015da12f74f26bc8f81c480b0a8a9d",
    },
    {
        "label": "original_m1_runner_726",
        "revision": "726b2dc6002be778ece01db59e2b5a62096f75ba",
        "path": "scripts/splicebert_acceptor_coupling/run_m1_donor_class.py",
        "size_bytes": 10817,
        "sha256": "2b5cffed3ae7161ac79786f050230f87b5203b22eb5d904f2e08e83c67159fea",
    },
    {
        "label": "support_aware_matcher_805e",
        "revision": "805e47ce96f12f98d09626e21bdc1c98a6f5a432",
        "path": "src/gfmcd/splicing/donor_class.py",
        "size_bytes": 11860,
        "sha256": "5e53fa496dd50f0121ed74ddcc50e4a5d45e4bd8c0dd8425b3cd4b4b1a9c159c",
    },
    {
        "label": "support_aware_m1_runner_805e",
        "revision": "805e47ce96f12f98d09626e21bdc1c98a6f5a432",
        "path": "scripts/splicebert_acceptor_coupling/run_m1_donor_class.py",
        "size_bytes": 11274,
        "sha256": "82fd78badd3924ef6e59dc093e510b338e3a5bc0838ece843321dce52301d699",
    },
    {
        "label": "m2m3_runner_805e",
        "revision": "805e47ce96f12f98d09626e21bdc1c98a6f5a432",
        "path": "scripts/splicebert_acceptor_coupling/run_m2m3_donor_class_dissection.py",
        "size_bytes": 10842,
        "sha256": "f4979b9797667a51e9e5171ab9ab9d127587a6b2ad0807125009bb1e9172e288",
    },
    {
        "label": "m3b_runner_805e",
        "revision": "805e47ce96f12f98d09626e21bdc1c98a6f5a432",
        "path": "scripts/splicebert_acceptor_coupling/run_m3b_l0h7_mediation.py",
        "size_bytes": 11603,
        "sha256": "384c01735c59819eb086a64581f4dee318b11592dc49679d81f276d25e158ef6",
    },
    {
        "label": "all_head_runner_bea3",
        "revision": "bea3f5fe64803ce76f77350d3b95701b2de987a8",
        "path": "scripts/splicebert_acceptor_coupling/run_m2m3_donor_class_dissection.py",
        "size_bytes": 11278,
        "sha256": "b1c344c6a33f476673fff99946e837fb29b7bdd9b1d004eb2f8ce759303dd925",
    },
)

# These pin the reviewed external bytes.  A later exploratory run must expand the registry
# under a new artifact revision; it must not silently change this result.
EXPECTED_SOURCE_SHA256 = {
    "m1_development.json": "fc913fa08448d756f85a8437b1939453e4867bcee80db21e17537b2f7a6db38a",
    "m1_development.manifest.json": "35d7770b4ad022b33614e4d04f80c29295d77d6eafb722c11b1d56696e529ff6",
    "m1_development_k1.json": "4e64c112c84db8aff7ab30cc1d6252dc4b7be5da476821bed8050ed5974e476d",
    "m1_development_k1.manifest.json": "f0e39da6cb0c70b267e3ea12f63e250f7bd754f0bbbd6ded6e24f3e18553e03d",
    "m2m3_development.json": "02b18537116058b8a46022166038dfa1aac6c7a579073cba309dbbb7c020f55a",
    "m2m3_development.manifest.json": "c8f1891aa7bc6940cff0fe30c84022536588c91c16ae3d5421303c713f9b9f40",
    "m3b_development.json": "10c14987c2454026522fbf6aa86a05839f0336aced2e3e29413173611fc6e639",
    "m3b_development.manifest.json": "a826c5fb52aff4522534220f002e7fe7e71a8f1235600c384a0f087b6dd05c9c",
    "m1_v2.json": "33d8657608b76ed2338d75bcc3882a2193b045cab6bd454529f523c61f198f99",
    "m1_v2.manifest.json": "b9e35673c343109a424aab0df10b51455078645aee35d5b29c309e17f08696e1",
    "m2m3_v2.json": "6fcba8c7d217fbc19c9bcc294aa479162c3a6874997c4f7896727e896ba5c340",
    "m2m3_v2.manifest.json": "c4ff664031ac83e03bc024efd38ca95f0188c62649a946ea4d511a0ba57fa1f4",
    "m3b_v2.json": "c04742c034c60dd9e61a0add09e820ff2ab19b1948ff1a899cd691aca40b0e6c",
    "m3b_v2.manifest.json": "d3e348af67b4ccd87e02eb1bc82cc57c6895668db502678cd434884d1986edf6",
    "m2m3_allheads.json": "f329e5b9e9941cbd49fd36fbbd25d3bae35b8a7793a10ccf795f24ac1f08c2eb",
    "m2m3_allheads.manifest.json": "4c4b303e4f4690bd654ec6a9f5ebda61f1fb87ebbac8cb993664954f458c4647",
    "claude_session_transcript.jsonl": "924e011378f4105c1017aeb1218b38c9c0f998fbd627208261b8841367a40409",
    "head_screen_clean_alt_comparison_v1.json": "04fec5998ba3d1dd17df27fb344100ad1c43ecd1e8e61338247a109ad223c8a6",
    "head_screen_clean_alt_comparison_v1.manifest.json": "8d1ca99841252112479ea8d67d7e853143a57393f41d87242970d3d03e6e5f93",
    "gencode.v50.annotation.gtf.gz": "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453",
    "hg38.fa": "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51",
    "hg38.fa.fai": "3b425de206296a5c8053023fa5ca61da43cfe78c1737c12e58c83367c7e83c21",
}
EXPECTED_HEAD_SHARD_INVENTORY_SHA256 = (
    "ecdedf3dd9578b9c1c66201a127ce39811e03471d5cdf0e5acd927376579b411"
)

EXPECTED_IDENTITY = {
    "claude_legacy_rows": 787,
    "claude_legacy_genes": 585,
    "claude_rows": 1057,
    "claude_genes": 794,
    "claude_row_ids_sha256": "bbd2b11d22142d7f1592226a696247d3f7140c2ee826374e0c259fbff1446b23",
    "claude_gene_ids_sha256": "3eb45c7037e90e6a87ce08c92d535315b8496f9f8a242a9c353f19e62bc1fbf5",
    "head_rows": 400,
    "head_genes": 360,
    "head_row_ids_sha256": "605f47087a95aaf02aa977e46abdacf8f15bc9abc65f238d8abf00cff263380e",
    "head_gene_ids_sha256": "02e335788ed2b08d5b3dad5d2e5e4ee4b91a9c8dcc51bf0772414880abec0f92",
    "source_overlap_genes": 42,
    "registry_genes": 1112,
    "registry_gene_ids_sha256": "378aa6b5b9408b6bb3311b883cc9ba984678dc287f442071a82dc8a9071efc09",
    "eligible_rows": 14496,
    "eligible_genes": 4843,
    "eligible_gene_ids_sha256": "3a2627e89280a02aacf3087ca2fc99db30e58634cc508781416efe59e9a8cefe",
    "exposed_eligible_rows": 4584,
    "exposed_eligible_genes": 934,
    "exposed_eligible_gene_ids_sha256": "2201c734c66c31f3272369827c201550f229edbb6a7577c5125adf7c513db851",
    "fresh_rows": 9912,
    "fresh_genes": 3909,
    "fresh_gene_ids_sha256": "c41dbd0222f7a7364f9d8558e06854ad758228d8962b309dd42a54170c590cd4",
    "model_scored_chr8_chr9_rows": 0,
}


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(rendered).hexdigest()


def _hash_rank(salt: str, *values: Any) -> str:
    return hashlib.sha256(
        "|".join([salt, *(str(value) for value in values)]).encode()
    ).hexdigest()


def file_binding(path: str | Path, *, logical_name: str | None = None) -> dict[str, Any]:
    source = Path(path)
    return {
        "logical_name": logical_name or source.name,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def verify_binding(binding: Mapping[str, Any]) -> None:
    name = str(binding["logical_name"])
    expected = EXPECTED_SOURCE_SHA256.get(name)
    if expected is None:
        raise RuntimeError(f"no reviewed source hash is registered for {name}")
    if binding.get("sha256") != expected:
        raise RuntimeError(
            f"reviewed source bytes drifted for {name}: "
            f"observed={binding.get('sha256')} expected={expected}"
        )


def bind_reconstruction_git_blobs(worktree: str | Path) -> dict[str, Any]:
    """Bind committed source snapshots used to audit omitted-row reconstruction."""
    output: dict[str, Any] = {}
    for spec in RECONSTRUCTION_BLOB_SPECS:
        blob = subprocess.run(
            [
                "git",
                "show",
                f"{spec['revision']}:{spec['path']}",
            ],
            cwd=worktree,
            check=True,
            capture_output=True,
        ).stdout
        observed = {
            "revision": spec["revision"],
            "path": spec["path"],
            "size_bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        expected = {
            key: spec[key]
            for key in ("revision", "path", "size_bytes", "sha256")
        }
        if observed != expected:
            raise RuntimeError(
                f"Claude reconstruction blob drifted for {spec['label']}: "
                f"observed={observed} expected={expected}"
            )
        output[str(spec["label"])] = observed
    return output


def donor_motif(example: Any) -> str:
    return example.sequence[example.donor_pos : example.donor_pos + 2]


def _gap(example: Any) -> int:
    if example.next_acceptor_pos is None:
        raise ValueError(f"{example.example_id} has no intron-mate acceptor")
    return int(example.next_acceptor_pos - example.donor_pos)


def flank_key(example: Any) -> tuple[str, ...]:
    return tuple(
        example.sequence[example.donor_pos + offset] for offset in (-1, 2, 4)
    )


def isoform_bin(example: Any) -> str:
    n = int(example.n_isoforms_donor or 0)
    if n <= 2:
        return "iso<=2"
    if n <= 5:
        return "iso3-5"
    if n <= 11:
        return "iso6-11"
    return "iso12+"


def flank_and_support_key(example: Any) -> tuple[str, ...]:
    return flank_key(example) + (isoform_bin(example),)


@dataclass(frozen=True)
class MatchedCohort:
    treatment: tuple[Any, ...]
    comparators: tuple[Any, ...]
    unmatched_treatment: tuple[str, ...]
    k: int


def match_comparators(
    treatment: Sequence[Any],
    pool: Sequence[Any],
    *,
    k: int,
    key_fn: Callable[[Any], tuple[str, ...]] = flank_key,
) -> MatchedCohort:
    """Reproduce the outcome-blind Claude matcher without importing that branch."""
    if k < 1:
        raise ValueError("k must be positive")
    by_stratum: dict[tuple[str, ...], list[Any]] = defaultdict(list)
    for candidate in pool:
        by_stratum[key_fn(candidate)].append(candidate)
    for rows in by_stratum.values():
        rows.sort(key=lambda row: row.example_id)

    used: set[str] = set()
    kept_treatment: list[Any] = []
    kept_comparators: list[Any] = []
    unmatched: list[str] = []
    for example in sorted(treatment, key=lambda row: row.example_id):
        candidates = [
            candidate
            for candidate in by_stratum.get(key_fn(example), ())
            if candidate.example_id not in used
        ]
        if len(candidates) < k:
            unmatched.append(str(example.example_id))
            continue
        candidates.sort(
            key=lambda candidate: (
                abs(_gap(candidate) - _gap(example)),
                candidate.example_id,
            )
        )
        chosen = candidates[:k]
        used.update(str(candidate.example_id) for candidate in chosen)
        kept_treatment.append(example)
        kept_comparators.extend(chosen)
    return MatchedCohort(
        treatment=tuple(kept_treatment),
        comparators=tuple(kept_comparators),
        unmatched_treatment=tuple(unmatched),
        k=k,
    )


def donor_class_pool(examples: Iterable[Any], motif: str) -> list[Any]:
    return [
        example
        for example in examples
        if example.split == "train"
        and example.next_acceptor_pos is not None
        and donor_motif(example) == motif
        and GAP_MIN <= _gap(example) <= GAP_MAX
        and not planted_donor_motifs(
            example.sequence, example.donor_pos, MATCHED_OPERATOR
        )
    ]


def _cohort_summary(cohort: MatchedCohort) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = {
        str(example.example_id): example
        for example in (*cohort.treatment, *cohort.comparators)
    }
    row_ids = sorted(rows)
    genes = sorted({str(example.gene_id) for example in rows.values()})
    summary = {
        "treatment_rows": len(cohort.treatment),
        "comparator_rows": len(cohort.comparators),
        "unmatched_treatment_rows": len(cohort.unmatched_treatment),
        "unique_rows": len(row_ids),
        "unique_genes": len(genes),
        "row_ids_sha256": canonical_sha256(row_ids),
        "gene_ids_sha256": canonical_sha256(genes),
    }
    return summary, rows


def reconstruct_claude_source(examples: Sequence[Any]) -> dict[str, Any]:
    gc_rows = donor_class_pool(examples, "GC")
    gt_rows = donor_class_pool(examples, "GT")
    cohorts = {
        "flank_k3": match_comparators(gc_rows, gt_rows, k=3),
        "flank_k1": match_comparators(gc_rows, gt_rows, k=1),
        "flank_support_k1": match_comparators(
            gc_rows, gt_rows, k=1, key_fn=flank_and_support_key
        ),
        "flank_support_k3_805e": match_comparators(
            gc_rows, gt_rows, k=3, key_fn=flank_and_support_key
        ),
    }
    summaries: dict[str, Any] = {}
    row_maps: dict[str, dict[str, Any]] = {}
    for name, cohort in cohorts.items():
        summaries[name], row_maps[name] = _cohort_summary(cohort)

    legacy_rows = {
        row_id: example
        for name in ("flank_k3", "flank_k1", "flank_support_k1")
        for row_id, example in row_maps[name].items()
    }
    all_rows = {
        row_id: example
        for rows in row_maps.values()
        for row_id, example in rows.items()
    }

    def union_summary(rows: Mapping[str, Any]) -> dict[str, Any]:
        row_ids = sorted(rows)
        genes = sorted({str(example.gene_id) for example in rows.values()})
        return {
            "rows": len(row_ids),
            "genes": len(genes),
            "row_ids_sha256": canonical_sha256(row_ids),
            "gene_ids_sha256": canonical_sha256(genes),
        }

    return {
        "cohorts": summaries,
        "legacy_union": union_summary(legacy_rows),
        "union": union_summary(all_rows),
        "row_ids": sorted(all_rows),
        "gene_ids": sorted({str(example.gene_id) for example in all_rows.values()}),
        "row_map": all_rows,
        "flank_k1_row_map": row_maps["flank_k1"],
    }


def _artifact_arm_rows(record: Mapping[str, Any]) -> list[dict[str, str]]:
    return sorted(
        (
            {
                "arm": str(row["arm"]),
                "example_id": str(row["example_id"]),
                "gene_id": str(row["gene_id"]),
            }
            for row in record.get("rows", [])
        ),
        key=lambda row: (row["arm"], row["example_id"], row["gene_id"]),
    )


def _expected_flank_k1_arm_rows(source: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    cohort_rows = source["flank_k1_row_map"]
    treatment_ids = {
        row_id
        for row_id in cohort_rows
        if donor_motif(cohort_rows[row_id]) == "GC"
    }
    for row_id, example in cohort_rows.items():
        rows.append(
            {
                "arm": "GC" if row_id in treatment_ids else "GT",
                "example_id": row_id,
                "gene_id": str(example.gene_id),
            }
        )
    return sorted(rows, key=lambda row: (row["arm"], row["example_id"], row["gene_id"]))


def bind_and_validate_claude_artifacts(
    scratch_dir: str | Path,
    transcript_path: str | Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    scratch = Path(scratch_dir)
    bindings: dict[str, Any] = {}
    for name in (*CLAUDE_RESULT_NAMES, *CLAUDE_MANIFEST_NAMES):
        binding = file_binding(scratch / name)
        verify_binding(binding)
        bindings[name] = binding
    transcript = file_binding(
        transcript_path, logical_name="claude_session_transcript.jsonl"
    )
    verify_binding(transcript)

    records = {
        name: json.loads((scratch / name).read_text())
        for name in CLAUDE_RESULT_NAMES
    }
    manifest_records = {
        name: json.loads((scratch / name).read_text())
        for name in CLAUDE_MANIFEST_NAMES
    }
    semantic_bindings: dict[str, Any] = {}
    for result_name in CLAUDE_RESULT_NAMES:
        manifest_name = result_name.replace(".json", ".manifest.json")
        manifest = manifest_records[manifest_name]
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError(
                f"{manifest_name} must bind exactly one result output"
            )
        output = outputs[0]
        result_binding = bindings[result_name]
        command = manifest.get("command")
        if not isinstance(command, list) or "--out" not in command:
            raise RuntimeError(f"{manifest_name} command does not declare --out")
        command_out = command[command.index("--out") + 1]
        observed = {
            "manifest_name": manifest_name,
            "output_basename": Path(str(output.get("path"))).name,
            "output_sha256": output.get("sha256"),
            "output_size_bytes": output.get("size_bytes"),
            "command_out_basename": Path(str(command_out)).name,
        }
        expected = {
            "manifest_name": manifest_name,
            "output_basename": result_name,
            "output_sha256": result_binding["sha256"],
            "output_size_bytes": result_binding["size_bytes"],
            "command_out_basename": result_name,
        }
        if observed != expected:
            raise RuntimeError(
                f"{manifest_name} does not semantically bind {result_name}: "
                f"observed={observed} expected={expected}"
            )
        semantic_bindings[result_name] = {
            **observed,
            "recorded_git_commit": manifest.get("git_commit"),
            "recorded_git_dirty": manifest.get("git_dirty"),
            "semantic_binding_pass": True,
        }
    expected_rows = _expected_flank_k1_arm_rows(source)
    row_artifacts = (
        "m2m3_development.json",
        "m2m3_v2.json",
        "m2m3_allheads.json",
    )
    for name in row_artifacts:
        observed = _artifact_arm_rows(records[name])
        if observed != expected_rows:
            raise RuntimeError(f"{name} does not replay the reconstructed flank-k1 cohort")
    if len(records["m2m3_allheads.json"].get("per_head", {})) != 96:
        raise RuntimeError("m2m3_allheads.json is not the complete 96-head scan")

    expected_counts = {
        "m1_development.json": (3, 181, 543),
        "m1_development_k1.json": (1, 181, 181),
        "m1_v2.json": (3, 181, 543),
    }
    for name, (k, treatment, comparator) in expected_counts.items():
        cohort = records[name].get("cohort", {})
        observed = (
            cohort.get("k"),
            cohort.get("n_treatment"),
            cohort.get("n_comparator"),
        )
        if observed != (k, treatment, comparator):
            raise RuntimeError(f"{name} cohort metadata drifted: {observed}")
    if records["m1_v2.json"]["cohort"].get("n_comparator_genes") != 445:
        raise RuntimeError("m1_v2 support-aware comparator cohort identity drifted")
    for name in ("m3b_development.json", "m3b_v2.json"):
        if records[name].get("cohort", {}).get("n_pairs") != 181:
            raise RuntimeError(f"{name} no longer records the full 181-pair cohort")

    row_projection = _artifact_arm_rows(records["m2m3_allheads.json"])
    return {
        "artifact_bindings": bindings,
        "result_manifest_semantic_bindings": semantic_bindings,
        "transcript_binding": transcript,
        "all_head_cohort": {
            "heads": 96,
            "rows": len(row_projection),
            "arm_example_gene_projection_sha256": canonical_sha256(row_projection),
            "identical_to_reconstructed_flank_k1": True,
            "adds_new_exposure_ids": False,
        },
    }


def scan_head_screen_shards(shard_dir: str | Path) -> dict[str, Any]:
    directory = Path(shard_dir)
    paths = sorted(directory.glob("shard_*.json"), key=lambda path: path.name)
    inventory: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_chr8_chr9: list[str] = []
    for path in paths:
        binding = file_binding(path)
        inventory.append(
            {
                "name": path.name,
                "size_bytes": binding["size_bytes"],
                "sha256": binding["sha256"],
            }
        )
        example = json.loads(path.read_text())["example"]
        all_rows.append(example)
        if str(example.get("chrom")) in TEST_CHROMOSOMES:
            all_chr8_chr9.append(str(example["example_id"]))
    row_ids = sorted(str(row["example_id"]) for row in all_rows)
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("head-screen example IDs are not unique")
    genes = sorted({str(row["gene_id"]) for row in all_rows})

    def stratum(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        ids = sorted(str(row["example_id"]) for row in rows)
        gene_ids = sorted({str(row["gene_id"]) for row in rows})
        return {
            "rows": len(ids),
            "genes": len(gene_ids),
            "row_ids_sha256": canonical_sha256(ids),
            "gene_ids_sha256": canonical_sha256(gene_ids),
        }

    acceptor_rows = [row for row in all_rows if row.get("target_is_donor") is False]
    donor_rows = [row for row in all_rows if row.get("target_is_donor") is True]
    if len(acceptor_rows) + len(donor_rows) != len(all_rows):
        raise RuntimeError("every head-screen row must declare a boolean target_is_donor")
    acceptor_genes = {str(row["gene_id"]) for row in acceptor_rows}
    donor_genes = {str(row["gene_id"]) for row in donor_rows}
    return {
        "shards": len(paths),
        "shard_inventory_sha256": canonical_sha256(inventory),
        "shard_bindings": inventory,
        "rows": len(row_ids),
        "genes": len(genes),
        "row_ids_sha256": canonical_sha256(row_ids),
        "gene_ids_sha256": canonical_sha256(genes),
        "row_ids": row_ids,
        "gene_ids": genes,
        "endpoint_strata": {
            "masked_acceptor": stratum(acceptor_rows),
            "masked_donor": stratum(donor_rows),
            "gene_overlap": {
                "genes": len(acceptor_genes & donor_genes),
                "gene_ids_sha256": canonical_sha256(
                    sorted(acceptor_genes & donor_genes)
                ),
            },
        },
        "all_shard_chr8_chr9_row_ids": sorted(all_chr8_chr9),
    }


def r3c_train_eligible_rows(examples: Sequence[Any]) -> tuple[list[tuple[Any, int]], Counter]:
    exclusions: Counter[str] = Counter()
    output: list[tuple[Any, int]] = []
    for example in examples:
        if not (
            example.split == "train"
            and example.next_acceptor_pos is not None
            and donor_motif(example) == "GT"
            and GAP_MIN <= _gap(example) <= GAP_MAX
        ):
            continue
        decoy = matched_decoy_donor(example)
        if decoy is None:
            exclusions["missing_matched_decoy_donor"] += 1
            continue
        decoy = int(decoy)
        if decoy == int(example.donor_pos):
            exclusions["pseudo_donor_equals_paired_donor"] += 1
            continue
        if abs(decoy - int(example.donor_pos)) > DECOY_TOL:
            exclusions["donor_pseudo_motif_outside_distance_caliper"] += 1
            continue
        if not donor_alt_admissible(example, MOTIF_CLEAN_DONOR_ALT):
            exclusions["paired_donor_edit_inadmissible"] += 1
            continue
        if planted_donor_motifs(example.sequence, decoy, MOTIF_CLEAN_DONOR_ALT):
            exclusions["pseudo_donor_edit_plants_splice_motif"] += 1
            continue
        output.append((example, decoy))
    return output, exclusions


def one_hash_ranked_row_per_gene(
    rows: Sequence[tuple[Any, int]],
) -> list[tuple[Any, int]]:
    selected: dict[str, tuple[Any, int]] = {}
    ranks: dict[str, str] = {}
    for example, decoy in rows:
        gene = str(example.gene_id)
        rank = _hash_rank(ROW_RANK_SALT, gene, example.example_id, int(decoy))
        if gene not in ranks or rank < ranks[gene]:
            selected[gene] = (example, int(decoy))
            ranks[gene] = rank
    return sorted(
        selected.values(),
        key=lambda row: (
            _hash_rank(GENE_RANK_SALT, str(row[0].gene_id)),
            str(row[0].gene_id),
        ),
    )


def _row_record(row: tuple[Any, int]) -> dict[str, Any]:
    example, decoy = row
    return {
        "example_id": str(example.example_id),
        "gene_id": str(example.gene_id),
        "chrom": str(example.chrom),
        "decoy_donor": int(decoy),
    }


def capacity_record(
    eligible_rows: Sequence[tuple[Any, int]],
    registry_genes: Sequence[str],
    exclusions: Mapping[str, int],
) -> dict[str, Any]:
    registry = set(map(str, registry_genes))
    raw_records = sorted(
        (_row_record(row) for row in eligible_rows),
        key=lambda row: (row["example_id"], row["decoy_donor"]),
    )
    ranked = one_hash_ranked_row_per_gene(eligible_rows)
    exposed_ranked = [row for row in ranked if str(row[0].gene_id) in registry]
    fresh_ranked = [row for row in ranked if str(row[0].gene_id) not in registry]
    exposed_rows = [row for row in eligible_rows if str(row[0].gene_id) in registry]
    fresh_rows = [row for row in eligible_rows if str(row[0].gene_id) not in registry]
    eligible_genes = sorted({str(row[0].gene_id) for row in eligible_rows})
    exposed_genes = sorted({str(row[0].gene_id) for row in exposed_ranked})
    fresh_genes = sorted({str(row[0].gene_id) for row in fresh_ranked})

    def ranked_hash(rows: Sequence[tuple[Any, int]]) -> str:
        return canonical_sha256([_row_record(row) for row in rows])

    return {
        "eligible_rows": len(eligible_rows),
        "eligible_row_records_sha256": canonical_sha256(raw_records),
        "eligible_genes_one_row_each": len(ranked),
        "eligible_gene_ids_sha256": canonical_sha256(eligible_genes),
        "exclusions": dict(sorted((str(k), int(v)) for k, v in exclusions.items())),
        "exposed_development": {
            "rows_available": len(exposed_rows),
            "genes_one_row_each": len(exposed_ranked),
            "gene_ids_sha256": canonical_sha256(exposed_genes),
            "ordered_one_row_per_gene_sha256": ranked_hash(exposed_ranked),
        },
        "fresh_target": {
            "rows_available": len(fresh_rows),
            "genes_one_row_each": len(fresh_ranked),
            "gene_ids_sha256": canonical_sha256(fresh_genes),
            "ordered_one_row_per_gene_sha256": ranked_hash(fresh_ranked),
        },
        "one_row_per_gene_policy": {
            "row_rank_salt": ROW_RANK_SALT,
            "gene_rank_salt": GENE_RANK_SALT,
        },
    }


def load_train_dataset(
    gtf: str | Path,
    fasta: str | Path,
    *,
    loader: Callable[..., Sequence[Any]] = _dataset,
) -> list[Any]:
    dataset = list(
        loader(
            str(gtf),
            str(fasta),
            allowed_chromosomes=TRAIN_CHROMOSOMES,
        )
    )
    leaked = sorted(
        {str(example.chrom) for example in dataset if str(example.chrom) in FORBIDDEN_CHROMOSOMES}
    )
    if leaked:
        raise RuntimeError(f"forbidden chromosomes were materialized: {leaked}")
    return dataset


def _compact_bindings(bindings: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "size_bytes": int(binding["size_bytes"]),
            "sha256": str(binding["sha256"]),
        }
        for name, binding in sorted(bindings.items())
    }


def build_record(
    *,
    dataset: Sequence[Any],
    gtf: str | Path,
    fasta: str | Path,
    claude_scratch_dir: str | Path,
    claude_transcript: str | Path,
    claude_worktree: str | Path,
    head_shard_dir: str | Path,
    head_comparison: str | Path,
    head_comparison_manifest: str | Path,
) -> dict[str, Any]:
    claude = reconstruct_claude_source(dataset)
    claude_bindings = bind_and_validate_claude_artifacts(
        claude_scratch_dir, claude_transcript, claude
    )
    reconstruction_blobs = bind_reconstruction_git_blobs(claude_worktree)
    head = scan_head_screen_shards(head_shard_dir)
    if head["shard_inventory_sha256"] != EXPECTED_HEAD_SHARD_INVENTORY_SHA256:
        raise RuntimeError("the 400-shard head-screen inventory drifted")
    comparison_binding = file_binding(head_comparison)
    comparison_manifest_binding = file_binding(head_comparison_manifest)
    verify_binding(comparison_binding)
    verify_binding(comparison_manifest_binding)

    gtf_binding = file_binding(gtf)
    fasta_binding = file_binding(fasta)
    fasta_index_binding = file_binding(f"{fasta}.fai")
    verify_binding(gtf_binding)
    verify_binding(fasta_binding)
    verify_binding(fasta_index_binding)

    registry_genes = sorted(set(claude["gene_ids"]) | set(head["gene_ids"]))
    source_overlap = sorted(set(claude["gene_ids"]) & set(head["gene_ids"]))
    eligible_rows, exclusions = r3c_train_eligible_rows(dataset)
    capacity = capacity_record(eligible_rows, registry_genes, exclusions)
    claude_chr8_chr9 = sorted(
        row_id
        for row_id in claude["row_ids"]
        if row_id.startswith("chr8:") or row_id.startswith("chr9:")
    )
    known_chr8_chr9 = sorted(
        set(claude_chr8_chr9) | set(head["all_shard_chr8_chr9_row_ids"])
    )

    return {
        "schema": RESULT_SCHEMA,
        "artifact_revision": ARTIFACT_REVISION,
        "status": "REVIEWED_MODEL_FREE_EXPOSURE_SCOPE_REGISTRY",
        "scope_id": REGISTRY_SCOPE_ID,
        "gene_ids": registry_genes,
        "gene_ids_sha256": canonical_sha256(registry_genes),
        "counts": {
            "source_classes": 2,
            "genes_union": len(registry_genes),
            "genes_source_overlap": len(source_overlap),
            "model_scored_rows_source_sum": claude["union"]["rows"] + head["rows"],
        },
        "scope": {
            "included": (
                "direct inverse-acceptor candidate-selection/dissection cohorts plus all "
                "400 clean-alt rows whose donor and acceptor strata jointly selected reverse "
                "heads; the acceptor stratum is also exact masked-acceptor endpoint exposure"
            ),
            "freshness_label": "disjoint_from_two_registered_direct_exposure_classes",
            "not_included": (
                "HCA18 and the broader head-context Atlas train corpus, donor-circuit studies, "
                "and earlier R3 train scoring used for candidate generation"
            ),
            "global_freshness_claimed": False,
            "atlas_disclosure": (
                "R3d train genes can overlap HCA18/the broader Atlas, donor-circuit, and earlier "
                "R3 candidate-generation train scoring; absence from this registry establishes "
                "disjointness only from the two explicitly registered direct-exposure classes."
            ),
        },
        "sources": {
            "A_claude_inverse_acceptor": {
                "branch_landmarks": {
                    "seal_and_support_aware_fix": (
                        "805e47ce96f12f98d09626e21bdc1c98a6f5a432"
                    ),
                    "complete_all_head_scan": (
                        "bea3f5fe64803ce76f77350d3b95701b2de987a8"
                    ),
                },
                "selection": (
                    "union of flank k=3, flank k=1, support-aware k=1, and 805e "
                    "support-aware k=3 reconstructed cohorts"
                ),
                "legacy_pre_805e_union": claude["legacy_union"],
                "union": claude["union"],
                "cohorts": claude["cohorts"],
                "all_head_cohort": claude_bindings["all_head_cohort"],
                "artifact_bindings": _compact_bindings(
                    claude_bindings["artifact_bindings"]
                ),
                "result_manifest_semantic_bindings": claude_bindings[
                    "result_manifest_semantic_bindings"
                ],
                "reconstruction_git_blobs": reconstruction_blobs,
                "reconstruction_contract": {
                    "implementation": str(Path(__file__).relative_to(REPO)),
                    "implementation_sha256": sha256_file(Path(__file__)),
                    "validation": (
                        "all persisted row-bearing M2/M3 artifacts replay flank-k1 exactly; "
                        "omitted-row M1/M3 cohorts are reconstructed by the bound deterministic "
                        "matcher and runner snapshots"
                    ),
                    "dirty_source_limitation": (
                        "Claude manifests record dirty runs, so unrecoverable dirty bytes are not "
                        "claimed identical to committed blobs; transcript and result bytes are "
                        "separately pinned."
                    ),
                },
                "transcript_binding": {
                    "size_bytes": claude_bindings["transcript_binding"]["size_bytes"],
                    "sha256": claude_bindings["transcript_binding"]["sha256"],
                },
                "original_run_disclosure": (
                    "The original Claude runners materialized unrestricted chromosomes before "
                    "train filtering. This reconstruction restricts before FASTA fetch and "
                    "matches the selected train cohorts; it does not erase that instrument defect."
                ),
            },
            "B_head_screen_clean_alt_candidate_selection": {
                "selection": (
                    "all 400 shards because both target_is_donor strata contributed to "
                    "reverse-head candidate selection"
                ),
                "shards": head["shards"],
                "shard_inventory_sha256": head["shard_inventory_sha256"],
                "rows": head["rows"],
                "genes": head["genes"],
                "row_ids_sha256": head["row_ids_sha256"],
                "gene_ids_sha256": head["gene_ids_sha256"],
                "endpoint_strata": head["endpoint_strata"],
                "exact_masked_acceptor_endpoint_is_stratum_only": True,
                "comparison_artifact": {
                    "size_bytes": comparison_binding["size_bytes"],
                    "sha256": comparison_binding["sha256"],
                },
                "comparison_manifest": {
                    "size_bytes": comparison_manifest_binding["size_bytes"],
                    "sha256": comparison_manifest_binding["sha256"],
                },
            },
        },
        "source_gene_overlap": {
            "genes": len(source_overlap),
            "gene_ids_sha256": canonical_sha256(source_overlap),
        },
        "train_capacity": {
            "preconstruction_allowed_chromosomes": list(TRAIN_CHROMOSOMES),
            "forbidden_chromosomes": sorted(FORBIDDEN_CHROMOSOMES),
            "forbidden_chromosomes_materialized": False,
            **capacity,
        },
        "known_artifact_chr8_chr9_model_scored_check": {
            "scope": "the two registered source classes",
            "rows_found": len(known_chr8_chr9),
            "row_ids_sha256": canonical_sha256(known_chr8_chr9),
            "materialization_is_distinct_from_model_scoring": True,
        },
        "input_bindings": {
            "annotation": {
                "logical_name": gtf_binding["logical_name"],
                "size_bytes": gtf_binding["size_bytes"],
                "sha256": gtf_binding["sha256"],
            },
            "genome": {
                "logical_name": fasta_binding["logical_name"],
                "size_bytes": fasta_binding["size_bytes"],
                "sha256": fasta_binding["sha256"],
            },
            "genome_index": {
                "logical_name": fasta_index_binding["logical_name"],
                "size_bytes": fasta_index_binding["size_bytes"],
                "sha256": fasta_index_binding["sha256"],
            },
        },
        "generator": {
            "script": str(Path(__file__).relative_to(REPO)),
            "script_sha256": sha256_file(Path(__file__)),
            "model_loaded_or_run": False,
            "canonical_json": (
                "json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)"
            ),
        },
    }


def production_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    claude = record["sources"]["A_claude_inverse_acceptor"]
    head = record["sources"]["B_head_screen_clean_alt_candidate_selection"]
    capacity = record["train_capacity"]
    return {
        "claude_legacy_rows": claude["legacy_pre_805e_union"]["rows"],
        "claude_legacy_genes": claude["legacy_pre_805e_union"]["genes"],
        "claude_rows": claude["union"]["rows"],
        "claude_genes": claude["union"]["genes"],
        "claude_row_ids_sha256": claude["union"]["row_ids_sha256"],
        "claude_gene_ids_sha256": claude["union"]["gene_ids_sha256"],
        "head_rows": head["rows"],
        "head_genes": head["genes"],
        "head_row_ids_sha256": head["row_ids_sha256"],
        "head_gene_ids_sha256": head["gene_ids_sha256"],
        "source_overlap_genes": record["source_gene_overlap"]["genes"],
        "registry_genes": len(record["gene_ids"]),
        "registry_gene_ids_sha256": record["gene_ids_sha256"],
        "eligible_rows": capacity["eligible_rows"],
        "eligible_genes": capacity["eligible_genes_one_row_each"],
        "eligible_gene_ids_sha256": capacity["eligible_gene_ids_sha256"],
        "exposed_eligible_rows": capacity["exposed_development"]["rows_available"],
        "exposed_eligible_genes": capacity["exposed_development"]["genes_one_row_each"],
        "exposed_eligible_gene_ids_sha256": capacity["exposed_development"]["gene_ids_sha256"],
        "fresh_rows": capacity["fresh_target"]["rows_available"],
        "fresh_genes": capacity["fresh_target"]["genes_one_row_each"],
        "fresh_gene_ids_sha256": capacity["fresh_target"]["gene_ids_sha256"],
        "model_scored_chr8_chr9_rows": record[
            "known_artifact_chr8_chr9_model_scored_check"
        ]["rows_found"],
    }


def validate_production_identity(record: Mapping[str, Any]) -> None:
    observed = production_identity(record)
    if observed != EXPECTED_IDENTITY:
        raise RuntimeError(
            "R3d exposure registry identity drifted: "
            f"observed={observed} expected={EXPECTED_IDENTITY}"
        )
    if record["counts"]["source_classes"] != 2:
        raise RuntimeError("the reviewed registry must contain exactly two source classes")
    if record.get("scope_id") != REGISTRY_SCOPE_ID:
        raise RuntimeError("the registry scope identifier drifted")
    if record["gene_ids"] != sorted(set(record["gene_ids"])):
        raise RuntimeError("registry gene IDs must be sorted and unique")
    if record["gene_ids_sha256"] != canonical_sha256(record["gene_ids"]):
        raise RuntimeError("registry gene digest does not replay")


def write_immutable_json(path: str | Path, record: Mapping[str, Any]) -> None:
    """Atomically publish a new result and refuse to replace any existing bytes."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"refuse to overwrite immutable result: {target}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def repository_state(repo: str | Path = REPO) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return {"git_revision": revision, "git_dirty": bool(status.strip())}


def require_clean_repository_state(repo: str | Path = REPO) -> dict[str, Any]:
    """Capture a clean source revision before create-only outputs are published.

    The registry and its manifest are intentionally new, untracked files.  Measuring
    ``git status`` after writing the registry would therefore label every otherwise-clean
    generation dirty.  Callers must invoke this helper while both declared outputs are
    still absent, and must recheck the returned state immediately before publication.
    Existing tracked, staged, or unrelated untracked changes remain fatal.
    """
    state = repository_state(repo)
    if state["git_dirty"] is not False:
        raise RuntimeError(
            "R3d exposure-registry generation requires a clean source worktree "
            "before create-only output publication"
        )
    return state


def build_manifest(
    *,
    record: Mapping[str, Any],
    output_path: str | Path,
    head_shard_dir: str | Path,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    output = file_binding(output_path)
    output_logical = Path(output_path).expanduser()
    if output_logical.is_absolute():
        resolved_output = output_logical.resolve()
        try:
            output_logical = resolved_output.relative_to(REPO.resolve())
        except ValueError:
            # Clean generation may publish into a separate durable results worktree.
            # Preserve that exact absolute location so the sibling manifest remains
            # an unambiguous binding rather than inventing a non-resolving relative path.
            output_logical = resolved_output
    claude = record["sources"]["A_claude_inverse_acceptor"]
    head = record["sources"]["B_head_screen_clean_alt_candidate_selection"]
    head_inventory = scan_head_screen_shards(head_shard_dir)
    if head_inventory["shard_inventory_sha256"] != head["shard_inventory_sha256"]:
        raise RuntimeError("manifest head-shard inventory differs from the result")
    state = dict(source_state)
    revision = state.get("git_revision")
    if (
        state.get("git_dirty") is not False
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError("manifest requires a captured clean Git source revision")
    return {
        "schema": "gfmcd.r3d.exposed_gene_registry_manifest/1",
        "experiment": "r3d_train_exposure_registry_v1",
        "status": "REVIEWED_MODEL_FREE_EXPOSURE_SCOPE_REGISTRY",
        "command": [
            "uv",
            "run",
            "python",
            str(Path(__file__).relative_to(REPO)),
            "--gtf",
            "<bound:gencode.v50.annotation.gtf.gz>",
            "--fasta",
            "<bound:hg38.fa>",
            "--claude-scratch-dir",
            "<bound:claude-m1-m2-m3-scratch-snapshot>",
            "--claude-transcript",
            "<bound:claude_session_transcript.jsonl>",
            "--claude-worktree",
            "<bound:claude-reconstruction-git-blobs>",
            "--head-shard-dir",
            "<bound:head_screen_clean_alt-400-shards>",
            "--head-comparison",
            "<bound:head_screen_clean_alt_comparison_v1.json>",
            "--head-comparison-manifest",
            "<bound:head_screen_clean_alt_comparison_v1.manifest.json>",
            "--out",
            str(output_logical),
        ],
        "parameters": {
            "scope_id": REGISTRY_SCOPE_ID,
            "source_classes": 2,
            "source_a_cohorts": [
                "flank_k3",
                "flank_k1",
                "flank_support_k1",
                "flank_support_k3_805e",
            ],
            "source_b_shards": head["shards"],
            "source_b_candidate_selection_uses_both_endpoint_strata": True,
            "operator": MATCHED_OPERATOR,
            "gap_min": GAP_MIN,
            "gap_max": GAP_MAX,
            "decoy_tolerance": DECOY_TOL,
            "preconstruction_allowed_chromosomes": list(TRAIN_CHROMOSOMES),
            "forbidden_chromosomes": sorted(FORBIDDEN_CHROMOSOMES),
            "row_rank_salt": ROW_RANK_SALT,
            "gene_rank_salt": GENE_RANK_SALT,
        },
        "seeds": {
            "coupling_dataset_seed": CouplingConfig().seed,
            "matching": "deterministic; no random seed",
        },
        "code": {
            "builder": record["generator"]["script"],
            "builder_sha256": record["generator"]["script_sha256"],
            **state,
        },
        "inputs": {
            "annotation": record["input_bindings"]["annotation"],
            "genome": record["input_bindings"]["genome"],
            "genome_index": record["input_bindings"]["genome_index"],
            "claude_artifacts_and_manifests": claude["artifact_bindings"],
            "claude_transcript": claude["transcript_binding"],
            "claude_result_manifest_semantic_bindings": claude[
                "result_manifest_semantic_bindings"
            ],
            "claude_reconstruction_git_blobs": claude[
                "reconstruction_git_blobs"
            ],
            "head_screen_shards": {
                "files": head_inventory["shard_bindings"],
                "files_count": head_inventory["shards"],
                "inventory_sha256": head_inventory["shard_inventory_sha256"],
            },
            "head_screen_comparison_artifact": head["comparison_artifact"],
            "head_screen_comparison_manifest": head["comparison_manifest"],
        },
        "outputs": [
            {
                "path": str(output_logical),
                "role": "registry",
                "logical_name": output["logical_name"],
                "size_bytes": output["size_bytes"],
                "sha256": output["sha256"],
            }
        ],
        "notes": [
            "Model-free cohort reconstruction and hashing only; no checkpoint was loaded.",
            "Git cleanliness was captured and rechecked before create-only output publication.",
            "Registry disjointness is limited to the two named direct-exposure classes; it is not global train-data freshness.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--claude-scratch-dir", required=True)
    parser.add_argument("--claude-transcript", required=True)
    parser.add_argument("--claude-worktree", required=True)
    parser.add_argument("--head-shard-dir", required=True)
    parser.add_argument("--head-comparison", required=True)
    parser.add_argument("--head-comparison-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out", default=None)
    args = parser.parse_args()

    manifest_out = (
        Path(args.manifest_out)
        if args.manifest_out is not None
        else Path(args.out).with_suffix(".manifest.json")
    )
    existing = [path for path in (Path(args.out), manifest_out) if path.exists()]
    if existing:
        raise SystemExit(
            "refuse to overwrite immutable output(s): "
            + ", ".join(str(path) for path in existing)
        )
    source_state = require_clean_repository_state()

    dataset = load_train_dataset(args.gtf, args.fasta)
    record = build_record(
        dataset=dataset,
        gtf=args.gtf,
        fasta=args.fasta,
        claude_scratch_dir=args.claude_scratch_dir,
        claude_transcript=args.claude_transcript,
        claude_worktree=args.claude_worktree,
        head_shard_dir=args.head_shard_dir,
        head_comparison=args.head_comparison,
        head_comparison_manifest=args.head_comparison_manifest,
    )
    validate_production_identity(record)
    if repository_state() != source_state:
        raise RuntimeError(
            "repository source state changed during R3d exposure-registry construction"
        )
    write_immutable_json(args.out, record)
    manifest = build_manifest(
        record=record,
        output_path=args.out,
        head_shard_dir=args.head_shard_dir,
        source_state=source_state,
    )
    write_immutable_json(manifest_out, manifest)
    print(
        json.dumps(
            {
                "out": str(Path(args.out)),
                "manifest": str(manifest_out),
                "rows_screened": len(dataset),
                "registry_genes": len(record["gene_ids"]),
                "exposed_development_genes": record["train_capacity"]
                ["exposed_development"]["genes_one_row_each"],
                "fresh_target_genes": record["train_capacity"]["fresh_target"]
                ["genes_one_row_each"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
