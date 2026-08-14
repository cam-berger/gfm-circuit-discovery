# CLAUDE.md — agent entry point & project core

Durable orientation for an agent session: what the project is, how it reasons, how to run it. It carries
**no running status** — anything dated, numeric, or contested belongs to the files under
[Where the current state lives](#where-the-current-state-lives), which own it and stay current.

## Mission

Discover **how genomic foundation models (GFMs) implement biological computations** — move from a model
*behavior* to a *causally supported mechanism* with explicit controls and a clear stopping point for
interpretation. **SpliceBERT is the first worked example, not the boundary.** The point is a
transferable *method*, not a catalogue of activations. A **second** model is now in the tree —
[DNABERT-2 promoter](studies/dnabert_promoter/README.md) (`DBP*`) — and the first thing it transferred
was a *limit*: it independently reached the same "structured causal core inside a distributed
computation" shape that the donor arc reached at **C33**. No SpliceBERT number transfers to it, and it
licenses nothing about any `C*`, `HCA*`, `AC*` or `GFS*` row.

- Method & rules: [`README.md`](README.md), [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md),
  [`docs/EVIDENCE_LADDER.md`](docs/EVIDENCE_LADDER.md), [`docs/TRANSFER_GUIDE.md`](docs/TRANSFER_GUIDE.md).
- **Never promote evidence** across levels (P → QK → Attention → OV/mediation → Behaviour → Biology).
  Attribution ranks; **intervention decides**. A position-table feature ≠ used by attention ≠ a causal
  computation ≠ a biological mechanism.
- **Vocabulary is controlled** — read [`docs/NOMENCLATURE.md`](docs/NOMENCLATURE.md) before writing live
  prose: several natural-sounding labels are retired or withdrawn, and near-identical numbers name
  different objects at different evidence levels.

## The mechanism, in one diagram

Corrupting a donor's paired acceptor moves the native masked-donor margin, partner-specifically and
donor-specifically — the donor "knows" its intron mate. The donor-graph arc A0→A12 dissected that
behaviour into an address, a payload, and a delivery step:

```
ADDRESS   position-embedding geometry (predominantly relative offset + modest absolute; C32),
          supplied by the period-3 carrier read across the donor–acceptor span (C34) and routed
          through L3H1's attention pattern (C35) ⇒ L3H1 selects the acceptor, not its identity (C22)
PAYLOAD   acceptor-local state → L0H7/H12/H10 → MLP0 (mediates) → MLP1 → MLP2 (C29, C31)
          ⇒ an ACCEPTOR-LOCAL payload, NOT explained by bulk composition, whose SEMANTIC IDENTITY IS
            UNRESOLVED (C30, bounded by R22 — never call it "splice-site strength"; that is withdrawn)
BIND/DELIVER  address × payload at L3H1 → donor-side low-rank U_info → transporter heads → donor readout
FAITHFULNESS  IRREDUCIBLY DISTRIBUTED (C33): no knee in the held-out size curve — a structured causal
              core within an irreducibly distributed computation, not a sparse circuit; minimality unclaimed
```

The frozen one-sentence mechanism statement, its checkpoint caveats, and its scope limits belong to
[`MECHANISM.md`](MECHANISM.md) — quote from there, not from memory.

## Where the current state lives

| Question | File that owns the answer |
|---|---|
| What is claimed, in what words? | the five `CLAIMS.md` ledgers that carry rows: [donor circuit](studies/splicebert_partner_selection/CLAIMS.md) (`C*`, 45), [GTEx fine-tuning shift](studies/splicebert_gtex_finetune_shift/CLAIMS.md) (`GFS*`, 34), [Head–Context Atlas](studies/splicebert_head_context_atlas/CLAIMS.md) (`HCA*`, 21), [DNABERT-2 promoter](studies/dnabert_promoter/CLAIMS.md) (`DBP*`, 17 — the only **non-SpliceBERT** ledger; `DBP11`–`DBP17` are the fine-tuned extension), and [acceptor coupling](studies/splicebert_acceptor_coupling/CLAIMS.md) (`AC*`, 4 — the **reverse** direction; a masked *acceptor* readout, deliberately kept out of the donor ledger). A sixth, [positional boundary coherence](studies/splicebert_positional_boundary_coherence/CLAIMS.md), is deliberately **empty**: that study is a frozen design with no outcome |
| What was retracted or bounded, and why? | [`CORRECTIONS.md`](studies/splicebert_partner_selection/CORRECTIONS.md) — **R1–R47**. R1–R24 are table rows (some are ranges); R25–R47 are `##` sections later in the same file — scan both. The DNABERT-2 study keeps its own: [`DBPR1–DBPR11`](studies/dnabert_promoter/CORRECTIONS.md) — read those before quoting any `DBP*` prose, five of that arc's headline sentences are bounded or withdrawn there, and `DBPR11` bounds the fine-tuned Layer-10 endpoint to a band. |
| Why was an experiment designed that way? | [`DECISIONS_A0.md`](studies/splicebert_partner_selection/DECISIONS_A0.md) and each study's `DECISIONS*.md` |
| What is next, and what is deferred? | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| Was a preregistered gate actually enforced? | [`docs/PREREG_CONFORMANCE.md`](docs/PREREG_CONFORMANCE.md) — **read before citing any gate as enforced**; many runners departed from their frozen protocol. Covers the SpliceBERT donor/carrier arc only — `GFS*` gates are self-reported, not audited, the positional-boundary-coherence gates have never been executed, and the `DBP*` study **froze no preregistration at all** (DBPR8), so nothing in it may be cited as a preregistered result |
| Which artifact controls a claim? | [`results/RESULTS_INDEX.md`](results/RESULTS_INDEX.md) and `results/MANIFEST.csv` |

Cite by ID; let the ledger carry the number, or it drifts. Seven study directories are in the tree and
**five carry the record** ([study map](README.md#study-map)); one,
[positional boundary coherence](studies/splicebert_positional_boundary_coherence/README.md), is a frozen
design whose ledgers are empty by construction, and one,
[exon-edge compatibility](studies/splicebert_exon_edge_compatibility/README.md), is an archive remnant
whose claims live in the donor ledger as C43/C44. One record-carrying study is easy to miss because the
study map still omits it: **acceptor coupling** (`AC*`, the reverse direction — a masked *acceptor*
readout). Five earlier study directories left the working tree in
the 2026-07-31 cleanup and remain in Git history. A sixth arc, **exon-edge compatibility**, was
consolidated on 2026-08-04 into two donor-ledger rows (**C43** behaviour, **C44** mediation-unresolved)
plus twelve artifacts and two frozen protocols; its bulk stayed on `codex/intron-exon-phase` at
`fd26cc1b`, and R45 governs what may be said about it. The positional-boundary-coherence study is that
arc's sanctioned successor and inherits blacklists and geometry from it, never its signs or magnitudes.

## How work is done here (conventions — follow these)

1. **Prereg → smoke → confirmatory → REFLECTION → claim.** A frozen study-local prereg with falsifiable
   memorials, a `--smoke` (memorials only, val sealed), then the run. No conclusions until memorials pass.
2. **Memorials must be able to fail, and must be on the measurement path** (R7/R21). Prefer exact
   arithmetic anchors (e.g. `value_linearity`); flag forced-by-architecture "memorials" as such.
3. **Calibration for identification claims:** planted-positive (sensitivity) + permuted-label FPR
   (specificity) gate the run — they have caught decision-layer bugs before any verdict was read.
4. **Held-out** gene splits wherever selection or ranking is involved — but within-sample held-out is not
   sampling stability: effect-existence needs multiple seeded draws plus two-way dyadic inference (R22).
5. **Claim-ledger discipline:** keep `CLAIMS.md` compact; record corrections under an R-number instead of
   editing history silently. Report conditional causal effects as such, **never additive fractions**.
6. **Adversarial verification** of finished claims; it has caught real overstatements. Correct honestly.
7. **Results write a manifest** beside their output; never silently overwrite the scientific record.
8. **Repo is private until submission; commit and push only on the user's cadence.** Where code and
   studies go, and what a run must emit, is in [`AGENTS.md`](AGENTS.md).

## Environment, running & code layout

- **`uv` is canonical** ([`REPRODUCE.md`](REPRODUCE.md)); run from the repo root —
  `uv run python scripts/splicebert/run_<exp>.py ...`, and pass the extras on tests too
  (`uv run --extra splicebert --extra biology --extra dev pytest`; a bare `uv run pytest` fails).
- `maxentpy` 0.0.4 (github kepbod/maxentpy, not PyPI) supplies MaxEntScan; `scripts/crossmodel/` needs a
  **separate** environment by design; the conda env `gfm-thrust1` in old records is provenance only.
- **Model:** SpliceBERT.1024nt, post-LayerNorm BERT-MLM (6L × 16H, hidden 512), **absolute learned
  position embeddings** (P=1026), checkpoint sha256 `2ad91428…0149d`. Post-LN ⇒ QKV linear in the
  layer-input residual, which is what makes exact reconstruct-inject possible.
- **Second model:** DNABERT-2 (`quietflamingo/dnabert2-no-flashattention`, revision `813031b2…31601`,
  12 layers, BPE over nucleotides) behind `src/gfmcd/adapters/dnabert.py`, for the `DBP*` study only.
  Its adapter returns the **12 encoder-layer outputs with no embedding output**, so `hidden_states[i]`
  *is* encoder layer `i` — the opposite of the usual HuggingFace convention, and asserted in the adapter.
  Never carry a layer index between the two models.
- Model, FASTA, and GTF are not committed; many memorials skip without a local genome. Long runs go to
  the background; smoke-gate first.
- `scripts/splicebert/` holds the donor-circuit runners (`run_a*_*.py` = reconstruct-inject / scrub /
  interchange); each study owns its own `scripts/<study>/` tree and its own frozen `PREREGISTRATION_*.md`
  protocols (under `preregistrations/` for the donor study, at the study root for the other two — the
  `DBP*` study has **none**, see DBPR8). `src/gfmcd/{splicing,circuits,adapters}/` hold the assays,
  reusable primitives and model access; `src/gfmcd/dnabert/` holds the DNABERT-2 dataset and
  perturbation utilities.
- **Every result writes a sibling `*.manifest.json`.** Use `gfmcd.provenance.ExperimentManifest`, or for
  the DNABERT-2 study the wrapper `gfmcd.dnabert.write_run_manifest`, which is wired into all 15 canonical
  `scripts/dnabert_promoter/` runners plus `analyze_finetuned_layer_selection.py`. Two gaps remain, both
  disclosed rather than back-filled: ~104 frozen DNABERT-2 artifacts predate that wiring, and the 52
  fine-tuned artifacts (`DBP11`–`DBP16`) came from `notebooks/dnabert_finetuned_experiments.ipynb` rather
  than from runners, so they carry blank `script` *and* `manifest` columns truthfully (`DBPR11`). Porting
  that notebook to `scripts/dnabert_promoter/` is the gating item before any fine-tuned row is promoted.
  Never invent a provenance record for a run whose conditions were not captured.
