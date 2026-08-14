"""Shared configuration for the NTv3 promoter-circuit study.

Mirrors the structure of the DNABERT-2 promoter-mechanism notebooks
(``dnabert/01_Representation_baseline.ipynb`` through ``09_...``), applied
to Nucleotide Transformer v3 (``InstaDeepAI/NTv3_8M_pre``) instead of
DNABERT-2. This replaces the earlier NTv2 port (``ntv2-interpretability/``,
removed) -- NTv3 is architecturally unrelated to NTv2's flat ESM-style
encoder, so the port starts fresh rather than search-replacing.

NTv3 is a U-Net: a convolutional downsampling tower feeds a small
Transformer bottleneck, which a matched deconvolutional tower upsamples
back to single-nucleotide resolution (verified against the real model in
``ntv3/model.py`` -- see that module's docstring for the architecture
facts this config's constants depend on). Sample sizes are reduced from
the DNABERT-2 originals (2000/800/800 pilot, 189/189 GC cohort) because
these experiments run on a CPU-only environment; every script prints and
records the sizes it actually used so no result is silently smaller than
it looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = REPO_ROOT / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Gated on Hugging Face: the account running these scripts must accept the
# license at https://huggingface.co/InstaDeepAI/NTv3_8M_pre and have an
# HF_TOKEN (env var, or `huggingface-cli login`) available -- see README.md.
MODEL_NAME = "InstaDeepAI/NTv3_8M_pre"

GUE_BASE_URL = (
    "https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/prom_core_all"
)
GUE_FILES = {
    "train": f"{GUE_BASE_URL}/train.csv",
    "dev": f"{GUE_BASE_URL}/dev.csv",
    "test": f"{GUE_BASE_URL}/test.csv",
}

# Pilot cohort sizes. DNABERT-2 notebooks used 1000/400/400 per class
# (2000/800/800 total). Scaled down for CPU-only inference; see README for
# the explicit tradeoff this documents.
PILOT_PER_CLASS_TRAIN = 250
PILOT_PER_CLASS_DEV = 100
PILOT_PER_CLASS_TEST = 100

# NB6-9 GC/causal cohort size. DNABERT-2 used 189+189. Kept smaller here.
CAUSAL_COHORT_PER_CLASS = 80

N_BOOTSTRAP = 2000  # DNABERT-2 notebooks used 2000-5000; 2000 throughout here.
N_SHUFFLE_REPEATS = 10
N_LABEL_SHUFFLES = 20

# GUE prom_core_all rows are a fixed 70nt. NTv3's U-Net requires the input
# length to be a multiple of 2**num_downsamples (128 for this checkpoint:
# num_downsamples=7). PAD_LENGTH is the padded length actually fed to the
# model; the true 70nt window is centered inside it with 'N' on both sides
# (see ntv3.model.pad_to_multiple) rather than left-aligned, so the signal
# sits at a consistent relative position instead of the tower's edge.
#
# At PAD_LENGTH=128 the Transformer bottleneck collapses to a single token
# (128 / 2**7 == 1): fine for phase01/02, which only touch pooled
# hidden-state content or black-box behavioral scores, but it makes
# attention-head-level localization (phase03+) close to vacuous -- there is
# nothing for a single bottleneck token to attend to but itself. That is a
# real architectural fork from the NTv2 port (a flat 12-layer encoder with
# no such degeneracy) and is tracked as an open decision before phase03 is
# written; do not assume PAD_LENGTH stays 128 for the whole pipeline.
SEQ_LEN_MULTIPLE = 128
PAD_LENGTH = 128
MAX_SEQ_LEN = 70  # true GUE prom_core_all sequence length, before padding


@dataclass(frozen=True)
class RunMeta:
    """Attached to every results JSON so downstream phases and readers know
    exactly what was actually run, not just what the script intends."""

    phase: str
    model: str = MODEL_NAME
    seed: int = SEED
    pad_length: int = PAD_LENGTH
    notes: list[str] = field(default_factory=list)
