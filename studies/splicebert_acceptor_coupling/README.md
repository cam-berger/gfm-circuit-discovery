# SpliceBERT acceptor coupling — paired-boundary dependence read from the other end

## Why this study exists

`studies/splicebert_partner_selection` masks the **donor** and corrupts the intron-mate **acceptor**,
in every confirmatory and mechanistic result it contains. The A0→A12 arc decomposed that behaviour into
a positional address, an **acceptor-local** payload, and delivery to a **donor-side** readout. That
account is stated asymmetrically throughout — and until this study, **nothing in the record tested
whether the asymmetry is a property of the model or of the assay.**

This study reads the same intron from the other end:

| | mask | corrupt | quantity |
|---|---|---|---|
| donor study (C1) | `donor_pos` | partner acceptor `[p-2, p)` | `Y_D` |
| **here** | `next_acceptor_pos - 1` | partner donor `[d, d+2)` | `Y_A` |

Both readouts mask a `G` — the donor's is the `G` of its `GT`, the acceptor's the `G` of its `AG` — so
the native masked-LM margin `logit(G) − max(A, C, T)` is used unchanged. The reversal is the
substitution of one position list for another, and `tests/test_reverse_coupling_memorials.py` asserts
at source level that `CouplingTask.encode`/`margin` contain nothing donor-specific.

## The one thing to know before designing anything here

**The instrument does not mirror.** The acceptor-side corruption operator is validated alt-invariant
within ±25% (**C42**). The donor-side operator is **not**, and cannot be:

- R44 measured `seq[d-1] == "G"` at **81.0%** of validation donors (the 5'SS consensus). `CC` therefore
  plants a `GC` donor one base upstream, `TA` plants a **canonical `GT`** (95.5% on the atlas
  condition-B cohort), and `AA` plants an `AG` (62.0%). `AC` and `AT` are universally motif-clean;
  `AC` is the clean two-substitution operator and `AT` the clean one-substitution operator.
- `GT → AC` is **two** substitutions; `GT → AT` is **one** and leaves the `T` at `d+1` standing.

C42 does not transfer — its own row scopes it to the acceptor operator. Every runner here uses
`MOTIF_CLEAN_DONOR_ALT` and re-verifies admissibility per construct with `planted_donor_motifs`, never
from a tuple. Any comparison between the two directions must match substitution counts;
`reverse_coupling.matched_operators` exists so that mismatch cannot be introduced silently.

## Prior operator evidence and the archived-edge boundary

One earlier donor-operator sensitivity result is relevant to this study:

| source | where | what it found | why it does not settle the question |
|---|---|---|---|
| `AR1` arm `DONA` | `splicebert_partner_selection`, `ar1_corruption_operator_sensitivity.json` | corrupt donor, read masked partner acceptor; verdict **`NOT_INVARIANT`**; every alt with `z[1] == "T"` at ρ ≈ 0.62 | declared "secondary and descriptive"; **no specificity control**; `n = 190` / 171 genes drawn entirely from the **19%** of donors lacking the consensus `G` |

The archived exon-edge compatibility prose is not a second reciprocal-acceptor result and is not folded
into this study. R45 carried only C43 and C44: C43 is a bounded **masked-donor** register-preference result
in a chimeric crossed lattice, while C44 says its proposed period-3 implementation is instrument-unresolved.
Neither supplies a native masked-acceptor operator estimate, component selection rule, or symmetry premise.

## What is here

Every arm is **verification tier**: validation split throughout, **sealed chr8/9 never opened and never
spent**, so a confirmatory reverse-coupling claim still has a seal to spend if one is ever warranted.

| arm | question | outcome |
|---|---|---|
| **R1** ([prereg](preregistrations/PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md), frozen) | is there a reverse coupling, and is it specific? | `INSTRUMENT_UNRESOLVED` — from a power gate aimed at the effect rather than at the nulls. **Superseded; retained as provenance.** See `ACR1`. |
| **R1b** ([prereg](preregistrations/PREREGISTRATION_R1b_REVERSE_COUPLING_GATE.md), frozen) | the same, with `MDE80` gated on the control arms + three-draw verdict stability | bounded validation gate: the artifact says **`GO`**, but the frozen scope forbids claim promotion and the permutation memorial is row-wise rather than gene-wise (`ACR8`) → `AC1` |
| **R2** | how does reverse coupling respond when each donor base is destroyed? | bounded post-outcome operator screen: either base changes the endpoint, sub-additively, with larger `T`-only interventions → `AC2` |
| **R3** ([prereg](preregistrations/PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md), frozen) | which of the 96 heads are **shared** between the two directions and which are **direction-specific**? | instrument unresolved: raw mean-ablation values nominate candidates, but the nominal partition and rank identifications are retracted because the calibration path did not implement the frozen familywise decision (`ACR9`) → `AC4` |
| **R4** ([retired draft](preregistrations/PREREGISTRATION_R4_EDGE_LEVEL_CAUSAL_CORE.md)) | historical question: do the two directions share their **edges**, or only their **components**? | never frozen; its exploratory execution used a defective objective and did not implement the proposed partition; no claim |

R3 was intended to answer "same or different" as a **single paired estimand** rather than two screens:
both readouts are measured on the same exons in the same process, so `Δ(c) = frac_A(c) − frac_D(c)`
has cohort, distance, composition and gene structure cancelling inside every contrast. Its primary
estimand is **mean-ablation**, because the repository's two idioms disagree 45× on exactly the class of
component at issue (forward `L2H15`: `+0.0115` noising vs `+0.5146` ablation) — a noising-primary sweep
returns a false null for the addresser. **MLP writes were out of scope in R3.** More importantly, R3's
permutation null, planted positive, one-direction specificity check, unadjusted equivalence rule, missing
noising estimand, and mismatched reverse-health policy make the frozen decision `INCONCLUSIVE` (`ACR9`,
`ACR11`). Its raw head effects remain candidate-generating measurements only.

R4 did not close those gaps. Its protocol was never frozen and is now retired; the completed exploratory
run used a signed objective that rewarded over-recovery and never implemented the preregistered
uncertainty-aware edge partition. See [`DECISIONS_R4_EDGE.md`](DECISIONS_R4_EDGE.md). No R4 result may be
used to infer shared components, shared wiring, mediation, or a reverse circuit.

- [`CLAIMS.md`](CLAIMS.md) — the claim ledger (`AC*`). Separate from the donor study's `C*` ledger on
  purpose: acceptor reconstruction is a different behaviour from donor reconstruction, and giving it
  its own ledger keeps C1–C42 from silently acquiring reverse-direction rows.
- [`DECISIONS.md`](DECISIONS.md) — chronology, including both pilots and their numbers, and the
  2026-08-08 BP arc (what it costs to leave the short-intron band; creates no claim row).
- [`DECISIONS_R4_EDGE.md`](DECISIONS_R4_EDGE.md) — `D1`–`D11`, the edge-level R4 design record,
  including the two negative results about the instrument (`D10` signed objective, `D11`
  unimplemented partition).
- [`CORRECTIONS.md`](CORRECTIONS.md) — `ACR*`. Never delete a correction; record, don't rewrite.
- [`preregistrations/`](preregistrations/) — `R1`, `R1b`, and `R3` are **frozen**; `R4` is a
  **retired draft that was never frozen**.
- `scripts/splicebert_acceptor_coupling/` — runners. `src/gfmcd/splicing/reverse_coupling.py` — the
  reusable readout. `results/splicebert_acceptor_coupling/` — artifacts, each with a manifest.

## Scope, standing

AC1–AC4 use one checkpoint (`SpliceBERT.1024nt`), canonical internal short introns, both strands, and the
validation split. AC1 is a bounded behavior gate, AC2 a bounded post-outcome operator screen, AC3 is
retracted, and AC4 is an instrument-unresolved candidate screen. They are model-internal, and **no
external biological assay enters those records.** Nothing here is a claim about splicing, spliceosome
biochemistry, or transcript abundance. Evidence never promotes across levels; read
[`../../docs/EVIDENCE_LADDER.md`](../../docs/EVIDENCE_LADDER.md) and
[`../../docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md) before writing prose here.
