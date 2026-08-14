# SpliceBERT positional-proxy modifiability — claim ledger

Point-of-contact record for the strongest wording currently licensed. One claim, one evidence level, one
controlling artifact set, one scope boundary per row. Status vocabulary matches the donor ledger:
**established** / **bounded** / **open** / **retracted** / **closed**.

## Live claims

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| PPM1 | **On the audited eligible census, no fresh, globally gene-disjoint real-module panel can satisfy the single-checkpoint `Ψ` endpoint, the composition-balance requirement, and the `0.05` resolution requirement together.** `Ψ` needs **718 blocks** against an exactly max-flow-feasible ceiling of **726** and a usable ceiling of **544** at a 25% assignment margin. The binding cause is structural: donor and acceptor pools overlap by **1869/1370/800 genes at phases 0/1/2**, so global disjointness spends each gene once. | **model-free construction/capacity finding — NOT behavioural, NOT biological.** No SpliceBERT checkpoint was loaded; no cohort was selected; no block was assembled; no cell was materialised. | `results/splicebert_positional_clock/real_panel_capacity_v1.json` (+manifest), produced by `scripts/splicebert_positional_clock/run_real_panel_capacity.py`; independently audited by `results/splicebert_positional_clock/real_panel_capacity_v1.audit.json` (+manifest) via `scripts/splicebert_positional_clock/audit_real_panel_capacity.py` — **`AUDIT_PASS`, 55/55 checks**, importing no verdict function from the producer or from any `gfmcd.splicing` module (the one declared shared low-level primitive is `networkx`'s max-flow solver; see the audit artifact's `independence` block) | **Scoped to standalone single-checkpoint `Ψ` only.** This does **NOT** generalize to the paired `ΔΨ = Ψ_L2 − Ψ_L1` endpoint, which remains **pilot-dependent and unresolved**: the same-block L1/L2 correlation a paired design would need (`0.621` at `N=544`, `0.494` at `N=726`, using the plain `{0,1}` lattice variance `(2 − ρ)·V9` — **not** the raw 3×3 `V9`, which would understate it as `0.382`/`0.175`) is not `C43`'s within-block `ρ` and is not estimable from anything committed to this record; it requires a seed-level pilot. `ρ = 0.368264` is a **point estimate with no interval**, from **one cohort**, under an approximate exchangeability assumption across cells within a compatibility group (cells differ by phase pair) — a design diagnostic, **not** a frozen sizing input. Every filter that could lift supply is load-bearing (freshness, phase-annotation safety, construct geometry: the three largest sole-reason exclusions among 282,057 total — `freshness_registry_gene`, `module:ValueError:native donor lacks the frozen health-window support`, `comprehensive_local_coding_register_conflict` — are each necessary for a panel that reads annotated register), so **no relaxation is available** to change this verdict. The controlling artifacts are **not clean-checkout re-executable**: the 2.34 GB candidate census and the 54 MB `C43` replication cells file both live outside this repository and are untracked; the audit only runs on a machine that already has both local files. |

## What is NOT claimed

- **No paired `ΔΨ` verdict.** PPM1 is silent on whether a `ΔΨ` transfer panel is constructible; that
  question is open and gated on a same-block L1/L2 covariance pilot this record does not contain.
- **No behavioural or biological content.** No checkpoint was loaded anywhere in the controlling
  artifacts. This is a gene-pool supply/demand and statistical-power finding about a **candidate
  construction design**, not a statement about SpliceBERT or about splicing.
- **No frozen sizing input.** `ρ = 0.368264` is a single-cohort point estimate under an approximate
  exchangeability assumption. A later, better-powered `ρ` estimate could move the block counts in this
  row; PPM1 would then need a superseding row, not a silent edit.
- **No claim about a differently-filtered or differently-designed panel.** PPM1 is scoped to the audited
  eligible census under the filters `real_panel_capacity_v1.json` actually applied (freshness, phase
  annotation safety, construct geometry, the plain `{0,1}` lattice). A materially different design (e.g.
  a synthetic module panel, or a relaxed geometry) is out of scope and would need its own construction
  and its own audit.

## Interpretation guardrails

- **`AUDIT_PASS` means the reported numbers reproduce independently, not that the underlying design
  choice is the only sane one.** The audit recomputes every B1–B5 quantity in
  `real_panel_capacity_v1.json` from the same raw inputs on its own code path (own census streamer, own
  slot-bucketing filter, own hashing, own sign-free variance decomposition, own power formulas, own
  max-flow feasibility call); it does not re-derive or bless the frozen `0.05` resolution requirement,
  the composition-balance requirement, or the choice of the plain `{0,1}` lattice over the withdrawn
  rotated 2×2 lattice (`647`-block ceiling, recorded for comparison only) — those are design decisions
  made upstream of this row.
- **`[a, b]` is a confidence interval and nothing else.** Nothing in this row is in that notation; every
  number here is a point value, a count, or an exact max-flow result.
- **This closes standalone `Ψ`, not the study.** A future `ΔΨ` pilot that measures the same-block L1/L2
  correlation could open a materially different capacity question; PPM1 says nothing about its outcome
  either way.
