# CLAUDE.md — agent entry point & project core

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Durable orientation for an agent session: what the project is, how it reasons, how to run it. It carries
**no running status** — anything dated, numeric, or contested belongs to the files under
[Where the current state lives](#where-the-current-state-lives), which own it and stay current.

## Mission

Discover **how genomic foundation models (GFMs) implement biological computations** — move from a model
*behavior* to a *causally supported mechanism* with explicit controls and a clear stopping point for
interpretation. **SpliceBERT is the first worked example, not the boundary.** The point is a
transferable *method*, not a catalogue of activations.

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
| What is claimed, in what words? | the three `CLAIMS.md` ledgers that carry rows: [donor circuit](studies/splicebert_partner_selection/CLAIMS.md) (`C*`), [Head–Context Atlas](studies/splicebert_head_context_atlas/CLAIMS.md) (`HCA*`), [GTEx fine-tuning shift](studies/splicebert_gtex_finetune_shift/CLAIMS.md) (`GFS*`). A fourth, [positional boundary coherence](studies/splicebert_positional_boundary_coherence/CLAIMS.md), is deliberately **empty**: that study is a frozen design with no outcome |
| What was retracted or bounded, and why? | [`CORRECTIONS.md`](studies/splicebert_partner_selection/CORRECTIONS.md) — **R1–R46**. R1–R24 are table rows (some are ranges); R25–R46 are `##` sections later in the same file — scan both. |
| Why was an experiment designed that way? | [`DECISIONS_A0.md`](studies/splicebert_partner_selection/DECISIONS_A0.md) and each study's `DECISIONS*.md` |
| What is next, and what is deferred? | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| Was a preregistered gate actually enforced? | [`docs/PREREG_CONFORMANCE.md`](docs/PREREG_CONFORMANCE.md) — **read before citing any gate as enforced**; many runners departed from their frozen protocol. Covers the SpliceBERT donor/carrier arc only — `GFS*` gates are self-reported, not audited, and the positional-boundary-coherence gates have never been executed |
| Which artifact controls a claim? | [`results/RESULTS_INDEX.md`](results/RESULTS_INDEX.md) and `results/MANIFEST.csv` |

Cite by ID; let the ledger carry the number, or it drifts. Four study directories are in the tree and
**three carry the record** ([study map](README.md#study-map)); the fourth,
[positional boundary coherence](studies/splicebert_positional_boundary_coherence/README.md), is a frozen
design whose ledgers are empty by construction. Five earlier study directories left the working tree in
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
- Model, FASTA, and GTF are not committed; many memorials skip without a local genome. Long runs go to
  the background; smoke-gate first.
- `scripts/splicebert/` holds the donor-circuit runners (`run_a*_*.py` = reconstruct-inject / scrub /
  interchange); each study owns its own `scripts/<study>/` tree and its own frozen `PREREGISTRATION_*.md`
  protocols (under `preregistrations/` for the donor study, at the study root for the other two).
  `src/gfmcd/{splicing,circuits,adapters}/` hold the assays, reusable primitives and model access.
