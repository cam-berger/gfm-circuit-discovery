"""Open a model by spec: one place that locates, verifies, and loads every checkpoint.

Before this module, 18 scripts called ``from_pretrained`` directly, 16 re-checked the 1024nt
checkpoint hash by hand, the SpliceBERT path was hard-coded 23 times and DNABERT-2's revision 12,
and manifests recorded model identity under at least five different keys.

    ModelSpec(name, family, source, revision, files, layer_convention, cites)
    open_model(spec, device=None, *, attention=False) -> ModelHandle(adapter, spec, identity, ...)

Local checkpoints are found in the checkout, then in each ``GFMCD_DATA_ROOTS`` entry, and every
listed file must match its sha256 before anything loads. Hub checkpoints are pinned by revision,
and the files actually loaded are hashed into ``identity``. ``attention=True`` loads eager attention,
because SDPA silently returns all-zero attention weights. ``handle.identity`` is exactly the
``model=`` argument of :func:`gfmcd.record.write`. The adapters are the pinned ones, unedited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Literal, Mapping

from gfmcd.provenance import sha256_file

REPO = Path(__file__).resolve().parents[2]
Family = Literal["splicebert", "dnabert2"]
# "embeddings_first": hidden_states[0] is the embedding output (HF convention, SpliceBERT).
# "encoder_only": hidden_states[i] IS encoder layer i (the DNABERT-2 adapter's convention).
LayerConvention = Literal["embeddings_first", "encoder_only"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: Family
    source: str  # repo-relative checkpoint directory, or a Hugging Face repo id
    revision: str | None = None
    files: Mapping[str, str] = field(default_factory=dict)  # filename -> sha256
    layer_convention: LayerConvention = "embeddings_first"
    cites: tuple[str, ...] = ()  # committed artifacts that record these hashes

    def __post_init__(self) -> None:
        if self.family not in ("splicebert", "dnabert2"):
            raise ValueError(f"unknown model family {self.family!r}")
        if self.layer_convention not in ("embeddings_first", "encoder_only"):
            raise ValueError(f"unknown layer convention {self.layer_convention!r}")
        if not self.files and not self.revision:
            raise ValueError(f"{self.name}: a spec needs file hashes or a pinned revision")


_ARCHIVE = "data/models/splicebert_official_archive/extracted/models"

SPLICEBERT_1024 = ModelSpec(
    name="SpliceBERT.1024nt",
    family="splicebert",
    source=f"{_ARCHIVE}/SpliceBERT.1024nt",
    files={
        "config.json": "51ea51e42951f18cc1d3455d7111fe0a41cebad476ba8314afc161a0e956303e",
        "pytorch_model.bin": "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d",
        "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
        "tokenizer.json": "b380c8c93ba20313c0c0239b8c70f0c67253028ca76cfda4f55b46049abe4fa3",
        "tokenizer_config.json": "a8f9a3828553689c1cb215689fc82af7551c1bb6ba642606612e834fda0c607c",
        "vocab.txt": "ba67bacb61d3de7d57e11e04ba59020a2cc67dbedbd9ae65715117b9fd4f30bc",
    },
    cites=("studies/splicebert_register_operator_replication/protocol_freeze_v2.json",),
)
SPLICEBERT_510 = ModelSpec(
    name="SpliceBERT.510nt",
    family="splicebert",
    source=f"{_ARCHIVE}/SpliceBERT.510nt",
    files={
        "config.json": "0e0daa1f987decd945cd76bc43a3ee4686e3715c470cb08fd2bf2f5c5296b4d8",
        "pytorch_model.bin": "b3a48c768a029dc6291cde8085c0bc02d40a6d03007b0535e6788573e2ea7808",
    },
    cites=("results/origin-siblings/canonical/sibling_validate.manifest.json",),
)
SPLICEBERT_HUMAN_510 = ModelSpec(
    name="SpliceBERT-human.510nt",
    family="splicebert",
    source=f"{_ARCHIVE}/SpliceBERT-human.510nt",
    files={
        "config.json": "9f3fb3b5f5a500b0aa600b063ecb78f1aefbe05a8bd9fab1843cb5894e70f048",
        "pytorch_model.bin": "7d9e7b6d53f2e42efbd19de2ad01c195244c138bff1e4468334294e3a62cae90",
    },
    cites=("results/origin-siblings/canonical/sibling_validate.manifest.json",),
)
DNABERT2 = ModelSpec(
    name="DNABERT-2 (no flash attention)",
    family="dnabert2",
    source="quietflamingo/dnabert2-no-flashattention",
    revision="813031b2bf86d9e960a027e2734c908009f31601",
    layer_convention="encoder_only",
    cites=("results/dnabert_promoter/behavioral/probe_results/validation_full_summary.json",),
)
REGISTRY = {spec.name: spec for spec in (SPLICEBERT_1024, SPLICEBERT_510, SPLICEBERT_HUMAN_510, DNABERT2)}


@dataclass(frozen=True)
class ModelHandle:
    adapter: Any  # SpliceBertAdapter or DnaBertAdapter, unedited
    spec: ModelSpec
    identity: dict  # {name, revision, files}: pass as gfmcd.record.write(model=...)
    attention: bool
    path: Path | None


def open_model(spec: ModelSpec, device: str | None = None, *, attention: bool = False) -> ModelHandle:
    """Locate, verify, and load ``spec``. Raises before loading if any listed file mismatches."""
    if spec.family == "splicebert":
        directory = locate(spec)
        from gfmcd.adapters.splicebert import SpliceBertAdapter

        extra = {"attn_implementation": "eager"} if attention else {}
        adapter = SpliceBertAdapter.from_pretrained(directory, device=device, local_files_only=True,
                                                    **extra)
        files = dict(spec.files)
    else:
        if attention:
            raise NotImplementedError("eager attention is not wired for DNABERT-2's remote code")
        from gfmcd.adapters.dnabert import DnaBertAdapter

        adapter = DnaBertAdapter.from_pretrained(spec.source, revision=spec.revision, device=device)
        directory, files = _hub_files(spec)
    identity = {"name": spec.name, "revision": spec.revision, "files": files}
    return ModelHandle(adapter, spec, identity, attention, directory)


def locate(spec: ModelSpec) -> Path:
    """The first candidate directory whose listed files all match: the checkout, then each data
    root (``root/source``, ``root/<source without data/>``, then any directory of the same name)."""
    if spec.family != "splicebert":
        raise ValueError("only local checkpoints are located; hub checkpoints load by revision")
    source = Path(spec.source)
    candidates = [REPO / source]
    for root in filter(None, os.environ.get("GFMCD_DATA_ROOTS", "").split(os.pathsep)):
        base = Path(root).expanduser()
        candidates += [base / source]
        if source.parts[0] == "data":
            candidates.append(base.joinpath(*source.parts[1:]))
        if base.is_dir():
            candidates += sorted(p for p in base.rglob(source.name) if p.is_dir())
    problems = []
    for directory in dict.fromkeys(candidates):  # de-duplicated, order kept
        if not directory.is_dir():
            continue
        mismatched = [name for name, sha in spec.files.items()
                      if not (directory / name).is_file() or _sha(directory / name) != sha]
        if not mismatched:
            return directory
        problems.append(f"{directory}: {', '.join(mismatched)}")
    detail = ("; mismatched or missing: " + " | ".join(problems)) if problems else ""
    raise FileNotFoundError(f"no verified copy of {spec.name} ({spec.source}){detail}. "
                            "Set GFMCD_DATA_ROOTS to a directory holding it.")


_HASHES: dict[tuple[str, int, int], str] = {}


def _sha(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        _HASHES[key] = sha256_file(path)
    return _HASHES[key]


def _hub_files(spec: ModelSpec) -> tuple[Path | None, dict[str, str]]:
    """Hash the cached snapshot actually loaded for a revision-pinned hub checkpoint."""
    try:
        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(spec.source, revision=spec.revision, local_files_only=True))
    except Exception:  # noqa: BLE001 - identity degrades to the revision pin, never to a guess
        return None, {}
    return directory, {p.name: _sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
