# [staged fix] notebook 09 inherits these from notebook 08's kernel
# and never creates them. Rebuilt here so 09 runs standalone.
#
#   dnabert2_local  synthetic package 08 registers in sys.modules (never on disk)
#   model_directory the pinned HF snapshot directory (08 calls it MODEL_LOCAL_DIR)
#   tokenizer       loaded from that snapshot
#   MAX_LENGTH      128, as set in 08
#   clean_sources   08 -> clean_source_cohort.csv
#   the subspaces and class targets -> 08's two npz archives
#
# Requires notebook 08 to have finished.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"
MODEL_REVISION = "813031b2bf86d9e960a027e2734c908009f31601"
MAX_LENGTH = 128

model_directory = Path(
    snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
    )
)

# 08 builds this package in memory only, so it does not survive into a new runtime.
PACKAGE_NAME = "dnabert2_local"

if PACKAGE_NAME not in sys.modules:
    package_module = type(sys)(PACKAGE_NAME)
    package_module.__path__ = [str(model_directory)]
    package_module.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package_module

tokenizer = AutoTokenizer.from_pretrained(
    model_directory,
    trust_remote_code=True,
)


# ------------------------------------------------------------
# Artifacts written by notebook 08
# ------------------------------------------------------------

CLEAN_COHORT_PATH = NOTEBOOK8_DIR / "clean_source_cohort.csv"
TARGET_ARCHIVE_PATH = NOTEBOOK8_DIR / "class_coordinate_targets.npz"
FINALIZED_SUBSPACE_PATH = NOTEBOOK8_DIR / "finalized_intervention_subspaces.npz"

for required_path in (
    CLEAN_COHORT_PATH,
    TARGET_ARCHIVE_PATH,
    FINALIZED_SUBSPACE_PATH,
):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Missing notebook 08 output: {required_path}\n"
            "Run notebook 08 to completion before this notebook."
        )

clean_sources = pd.read_csv(CLEAN_COHORT_PATH)

class_target_archive = np.load(TARGET_ARCHIVE_PATH)
finalized_subspaces = np.load(FINALIZED_SUBSPACE_PATH)

# 08: promoter_basis_rank1 = promoter_direction_unit_checked[np.newaxis, :]
promoter_basis_rank1 = np.asarray(
    finalized_subspaces["promoter_direction_unit"],
    dtype=np.float64,
)[np.newaxis, :]

gc_basis_rank4 = np.asarray(
    finalized_subspaces["gc_basis_rank4"],
    dtype=np.float64,
)

# Same convention this notebook uses for its GC targets, resolved the same way
# (float64, flattened) as resolve_class_target() does below.
promoter_direction_promoter_target = np.asarray(
    class_target_archive["promoter_direction_rank1_promoter_mean"],
    dtype=np.float64,
).reshape(-1)

promoter_direction_nonpromoter_target = np.asarray(
    class_target_archive["promoter_direction_rank1_non_promoter_mean"],
    dtype=np.float64,
).reshape(-1)


print("model_directory      :", model_directory)
print("dnabert2_local       :", PACKAGE_NAME in sys.modules)
print("tokenizer            :", type(tokenizer).__name__)
print("MAX_LENGTH           :", MAX_LENGTH)
print("clean_sources        :", clean_sources.shape)
print("archive keys         :", list(class_target_archive.files))
print("promoter_basis_rank1 :", promoter_basis_rank1.shape)
print("gc_basis_rank4       :", gc_basis_rank4.shape)
print(
    "promoter targets     :",
    promoter_direction_promoter_target.shape,
    promoter_direction_nonpromoter_target.shape,
)
