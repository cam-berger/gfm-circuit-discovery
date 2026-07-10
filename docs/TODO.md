# TODO — reproducibility and results

Work the repository needs so that a result can be regenerated from a clean checkout by someone who
did not produce it. Ordered by how much it blocks reproduction, not by effort. Each item says what is
wrong now, what "done" looks like, and where the relevant code lives.

Status legend: **[blocker]** stops a clean-checkout reproduction · **[correctness]** a result could be
wrong or read at the wrong place · **[hygiene]** does not change a number but makes the repo trustworthy.

---

## 1. Data provenance

### 1.1 hg38 / GENCODE are not in the repo and cannot be regenerated **[blocker]**
`data/genome.py` and `scripts/train_cell.py::pool` read `data/genomes/hg38.fa` and
`gencode.v50.basic.annotation.gtf.gz` to build the real-CDS cells (C and D). Neither file is present.
Only a subset of pre-generated windows exists as `data/cache/cell{C,D}_val_*.npz`, keyed by the exact
`(n, window, seed)` triple — request any other triple and the loader falls through to the missing
FASTA and raises `pyfaidx.FastaNotFoundError`. This is why the notebook 2×2 grid can build cells A and
B (synthetic, no genome needed) but not C and D.
- **Done when:** a `scripts/fetch_reference.py` downloads hg38 + the pinned GENCODE release to
  `data/genomes/` with a checksum check, *or* the full set of cached npz windows used by every
  committed result is saved as an artifact and documented. Record the GENCODE version and the
  `SPLIT_CHROMS` train/val/test split next to the fetcher.
- Until then, cells C/D in `notebooks/TEMPLATE_study_a_circuit.ipynb` §5b skip with a printed reason.

### 1.2 The pooled-oracle ceiling (D30) came from an uncommitted script **[correctness]**
`oracle.py`'s docstring notes the aggregate 0.9509 on real hg38 CDS at 240 nt "in a script that was
never committed." `scripts/pooled_oracle.py` now rebuilds and checks it — but the check needs 1.1's
data to run. Confirm the committed oracle reproduces the number once the genome is fetchable, and pin
the result in a test.

### 1.3 Cache files are inputs with no manifest **[hygiene]**
`data/cache/*.npz` are load-bearing (they stand in for the genome) but nothing records which script,
seed, and genome build produced each one. Add a `data/cache/MANIFEST.json` (or a `.meta` sidecar per
npz) with the generating parameters, so a stale cache is detectable.

---

## 2. Methodology correctness

### 2.1 Mechanism axes must not be read at a fixed layer on real CDS **[correctness]**
This is the repo's own hardest-won lesson (D34/D35): `signatures.py::LAYER = "L2.resid_post"` was
chosen when the only trained model was the uniform counter, which carries the anchor at L2. On real
hg38 CDS the anchor does not enter until L4, so **both** 2×2 axes read at L2 reported the wrong
mechanism for exactly the cells that produced a new claim.
- `lab.verdict_2x2` / `lab.cue_conflict_curve` currently read the anchor share at the `LAYER` probe.
  That is correct for the uniform aa-head models (cell A, the λ arms) but **wrong for cells C/D**.
- **Done when:** the y-axis for real-CDS cells uses the output-level readout
  (`scripts/phase_c_conflict_layers.py`) and the x-axis uses `peak_anchor_dependence`, not a fixed
  layer. Add a guard in `verdict_2x2` that refuses (or warns) when asked to read a mechanism off L2
  for a non-uniform pool. Tie this to 1.1 — it can only be validated once C/D data loads.

### 2.2 λ-sweep accuracy does not reach the committed 1.0000 at small N **[correctness]**
`docs/RESULTS.md` reports every λ arm at `aa_acc = 1.0000`; `lab.lambda_sweep` at `n≈384` shows some
arms below that (e.g. `nano_sense` ≈ 0.84, `nano_lam1` ≈ 0.75 on one seed). The mechanism axis (anchor
share) is the reported quantity and behaves correctly, but the accuracy column will confuse a reader
who compares to the table.
- **Done when:** the notebook λ-sweep documents the eval-set size that reproduces the table's 1.0000
  (or the sweep uses it), and the discrepancy is explained inline as a sample-size effect, not a
  regression.

### 2.3 Attribution vs intervention naming is easy to misread **[hygiene]**
`lab.attention_survey` ranks heads by attention; on `nano_uniform` the top-ranked head (L5H2) is
causally inert (Δloss ≈ 0) and the essential head (L1H3) ranks below it. This is intended and is the
"attribution ranks, intervention decides" lesson — but any doc or plot that presents the attention
ranking without the ablation table beside it invites the exact error the repo exists to warn against.
Keep the two adjacent everywhere (the notebook does; check the docs).

---

## 3. Environment & tooling

### 3.1 README still references the retired conda env **[blocker]**
`README.md` says `conda activate gfm-thrust1`; that environment does not exist. The repo is now
uv-managed (`uv.lock`, `.python-version` = 3.13, `UV_PROJECT_ENVIRONMENT=.venv/uv`, extras
`lab`/`dev`). Update the README quickstart to the uv flow: `uv sync --extra lab --extra dev`.
- **Done when:** a new user following the README alone reaches a working `import gfmcd` and a passing
  `pytest`.

### 3.2 Notebook execution needs a documented path **[hygiene]**
Standard `jupyter nbconvert --execute` binds a local ZMQ socket, which is blocked in the sandbox used
to build these notebooks; `scripts/execute_notebook.py` runs the cells in-process instead. A user on a
normal machine can just open the notebook with the `gfmcd-uv` kernel. Document both: the kernel is
installed via `python -m ipykernel install --sys-prefix --name gfmcd-uv`.

### 3.3 uv cache/python dirs must be redirected on restricted hosts **[hygiene]**
`uv` defaults to `~/.local/share/uv` and a repo-local `.uv_cache`, both of which failed here
(permission / `.git` file op). The working incantation set `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` to
`$TMPDIR`, and `UV_PYTHON_PREFERENCE=only-system`. Capture this in the README or a `Makefile` target so
the next person on a locked-down host does not rediscover it.

---

## 4. Testing & CI

### 4.1 No continuous integration **[hygiene]**
The suite (run `uv run pytest`; ~60 s locally) includes `tests/test_lab.py`. None
of it runs automatically. Add a CI workflow that runs `uv sync` + `pytest` on push. Mark the
genome-dependent tests with a skip marker keyed on `data/genomes/hg38.fa` so CI stays green without the
large files (the lab tests already skip when a checkpoint is absent).

### 4.2 Lock the headline numbers behind assertions **[correctness]**
The load-bearing results live in prose in `docs/RESULTS.md`. Promote the ones that gate the story into
tests so a refactor cannot silently move them: L1H3 ablation Δloss > 1.0 and L1H2 ≈ 0 (done in
`test_lab.py`), copy accuracy > 0.9 vs base rate < 0.35 (done), the pooled-oracle 0.9509 ceiling (needs
1.1), the 2×2 cell-A payoff (done), and each λ arm's mechanism verdict (add).

### 4.3 Determinism audit **[hygiene]**
Confirm every result-producing entry point takes an explicit seed and that MPS-vs-CPU does not move a
reported number past its last significant digit. Note any metric that is only stable to N digits.

---

## 5. Documentation

### 5.1 Two-track structure is not signposted **[hygiene]**
The repo holds two studies (reading-frame in `docs/`, repeat-copy in `branches/codex-repeat-induction/`)
that share code but no narrative. A top-level map — which docs belong to which track, and which
`checkpoints/` families back each — would save a newcomer the archaeology.

### 5.2 Retraction table needs to stay visible **[correctness]**
`docs/RESULTS.md §10` records the D24–D35 retractions (the mislabeled relays, the L2-probe error). Any
new write-up must not resurrect a retracted claim. Keep a one-line pointer to that table from every doc
that names a mechanism.

---

_Last updated: this reflects the state after the `gfmcd.lab` layer and notebooks were added. The
biggest single unblock is item 1.1 — it gates 1.2, 2.1, and half of §4._
