# B1 execution & integration handoff (2026-07-22)

> **Tier: T2 — provenance/handoff.** Dated execution/handoff record; not current canonical guidance.

Tier: **T2 provenance / handoff**. Not a claim ledger. For whoever runs B1 from the main checkout and
consolidates the carrier branch. Design of record: `preregistrations/PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR_V2.md`
(SPLIT-FREEZE); re-audit: `redteam/REDTEAM_SYNTHESIS_B1_V2.md`.

## Why this can't run in a worktree

The SpliceBERT.1024nt/510nt checkpoints and the genome/GENCODE data are **absent from the `.claude/worktrees/*`
checkouts** — `data/` there holds only `NUCLEOSOME_MAP.md`. All B1 model runs (and the `--smoke` gates) must be
launched from the **main checkout** (`/Users/cberg/Research/projects/gfm-circuit-discovery`) or a checkout with
`data/models` + `data/genomes` present. The drivers below are committed on the carrier branch, **syntax-clean
and API-verified but unit-untested** (no model here); the `--smoke` gate at the main checkout is their
validation contract.

Interpreter note: `conda run -n gfm-thrust1 …` errored with `permission denied: __conda_exe` in the review
harness; the env interpreter is at `/opt/homebrew/Caskroom/miniforge/base/envs/gfm-thrust1/bin/python`. Prefix
every command with `PYTHONHASHSEED=0`.

## The drivers (committed on `claude/carrier-ov-circuit-analysis-8acefd`)

| Driver | Arm | Status | Reuses |
|---|---|---|---|
| `scripts/splicebert/run_carrier_preflight_r.py` | Preflight R (the fork) | written, untested | `run_postsoftmax.py` m_of / profiles / deletion_audit / Level-A (verbatim) |
| `scripts/splicebert/run_carrier_p10_behaviour.py` | behavioural (FROZEN) | written, untested | `run_carrier_surgery.py` make/controls + `CouplingTask` |
| `run_carrier_ov_av.py` | OV/mediation | **not written — contingent on R_1024** | generalizes `run_p3_av_decomposition.py` (C35) |

## Run order (from the main checkout)

**1. Preflight R — the decisive fork.** Establishes whether any 1024nt head reaches post-softmax Level-A at
the ≈10.63 band (`R_1024`). C11's 8 readers are on the 510nt table, so this may be empty.

```
PYTHONHASHSEED=0 <py> scripts/splicebert/run_carrier_preflight_r.py --smoke   # mechanics: memorials + reader scan
PYTHONHASHSEED=0 <py> scripts/splicebert/run_carrier_preflight_r.py           # full -> results/phase-lattice/carrier_preflight_r.json
```
- `fork == "OV_ARM_LIVE"` (R_1024 non-empty) → proceed to step 3 (write + run the OV driver).
- `fork == "OV_ARM_ENDPOINT_B_behavioural_null_only"` (R_1024 = ∅) → **B1 OV arm is non-identifiable (Endpoint
  B), a pre-committed close.** B1 = the behavioural-null arm alone. Skip step 3.

**2. Behavioural arm (runs regardless of the fork; the FROZEN arm).** Coupling equivalence under ≈10.63
deletion; frozen prediction = NULL.

```
PYTHONHASHSEED=0 <py> scripts/splicebert/run_carrier_p10_behaviour.py --smoke  # M-del, M-ctrl, M-p3contrast, M-det
PYTHONHASHSEED=0 <py> scripts/splicebert/run_carrier_p10_behaviour.py --n 256  # -> carrier_p10_behaviour.json
```
- Sizing-pilot gate: freeze the equivalence only if `HW < SESOI_C − |point|` (= 0.026 at the seen `|point|≈0.024`;
  reachable at ρ>0.42). If `HW` is too wide → re-run at ~700+ genes or report one-sided; if `|point| ≥ 0.05` →
  **coupling POSITIVE**: report it and flag band 94–100 for re-selection as `carrier_surgery`/`dose`'s inert
  control (prereg §5 risk-2, the pre-committed dual branch).
- `M-p3contrast` (period-3 deletion collapses coupling, ratio ≤ 0.05) must pass, or the re-pointed instrument
  is not sensitive and nothing is interpreted.

**3. OV arm — ONLY if step 1 returned `OV_ARM_LIVE`.** Write `run_carrier_ov_av.py` (a generalization of
`run_p3_av_decomposition.py` over the `R_1024` heads read from `carrier_preflight_r.json`, restricted to L0
readers per must-fix #7), then smoke + run. Its estimand is the **simple single-row value main** `Mv_h = DI−DD`
(NOT the symmetric main, which re-imports ½·Δ_bind — the M-1a catch), projected onto `U_info`/coupling, with a
**covariance-matched** null, the period-3 L3H1 **router-negative** at the readout scale (`SESOI_v ≥ 0.02`), and
`PP-ovplant`. `Δ_bind` is reported, never interpreted (R23). This driver is deferred until R_1024 is known
because its heads, and its very necessity, come from step 1.

## Integration / consolidation (owner: Cam — git-destructive, touches live sessions)

The carrier branch is a **clean fast-forward** of master (ahead 2 — `2baf31e` B1–B5+HCA23, `72b032e` B1 v2 —
behind 0; merge-base *is* master). Ledger edits are append-only, so they compose.

1. **Rescue the review docs** — DONE (`cbf0c5a` on `p3-closure-confirmatory`; the owning session committed its
   own `docs/REVIEW_*`, `CLAIM_TRIAGE_*`, `OPEN_TRACKS_EXPANDED_*`).
2. **FF `master`** to `claude/carrier-ov-circuit-analysis-8acefd` (zero conflicts).
3. **Audit the unexamined branches** for anything worth keeping: `repo-structure-review` (+3),
   `dashboard-bookkeeping-cleanup` (+1), `ultracode-effort` (+2), `project-knowledge-quiz` (base). The carrier
   edits are append-only; these are unaudited.
4. **Prune merged worktrees** (the gitignored `.claude/worktrees/*` duplicates).
5. **Post-merge hygiene pass:** load `gfm-repo-hygiene` and run `audit_repo(<root>)` on the consolidated tree —
   one pass confirms the new B-prereg/result files carry manifests where needed, the results classify
   canonical-vs-development correctly, and no retired terms leaked in.

## Science status (verified)

- **B5 → CLOSED** (HCA23, CLOSE-AS-UNDERPOWERED, independently re-derived).
- **B4 → DEFERRED** (non-identifiable direct edge on a frozen post-LN checkpoint).
- **B1 → SPLIT-FROZEN**: behavioural arm frozen-runnable (driver written); OV arm design-frozen, driver
  contingent on Preflight R.
- **B2 / B3 → FIXABLE-THEN-RUNNABLE** (must-fixes documented; execution needs the main checkout).

No branch needs more red-teaming. What remains is execution (from the main checkout) + the FF/prune integration.
