# SpliceBERT acceptor coupling — the donor circuit read from the other end

> **Tier: T1 — canonical.** Current, authoritative. Read this.

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
  condition-B cohort), `AA` plants an `AG` (62.0%). Only `AC` plants nothing anywhere measured.
- `GT → AC` is **two** substitutions; `GT → AT` is **one** and leaves the `T` at `d+1` standing.

C42 does not transfer — its own row scopes it to the acceptor operator. Every runner here uses
`MOTIF_CLEAN_DONOR_ALT` and re-verifies admissibility per construct with `planted_donor_motifs`, never
from a tuple. Any comparison between the two directions must match substitution counts;
`reverse_coupling.matched_operators` exists so that mismatch cannot be introduced silently.

## Prior evidence this study folds in rather than re-deriving

Two reverse-direction results existed before this study and neither carried a claim row. They were
produced by different studies that never cite each other, and **they agree on something neither
states**: a donor destroyed with one substitution retains most of its effect.

| source | where | what it found | why it does not settle the question |
|---|---|---|---|
| `AR1` arm `DONA` | `splicebert_partner_selection`, `ar1_corruption_operator_sensitivity.json` | corrupt donor, read masked partner acceptor; verdict **`NOT_INVARIANT`**; every alt with `z[1] == "T"` at ρ ≈ 0.62 | declared "secondary and descriptive"; **no specificity control**; `n = 190` / 171 genes drawn entirely from the **19%** of donors lacking the consensus `G` |
| `E11` reciprocal coupling | `splicebert_exon_edge_compatibility` (branch `claude/acceptor-donor-reframe`, `b70ed2c5`) | verdict `RECIPROCAL_SYMMETRIC`, twelve gates pass, 256 sealed blocks | operator-bounded by its own disclosure: `Retention_A` is `+0.033` at two substitutions and `+0.218` at one; chimeric crossed-lattice geometry, not the native cohort; **no claim row**, its execution order puts an independent audit first |

## What is here

- [`preregistrations/PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md`](preregistrations/PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md)
  — frozen 2026-08-05. A **gate**, verification tier by construction: validation split, sealed chr8/9
  not opened, design established by two disclosed pilots, decision rule corrected after the second
  (§7.1). Its only job is to decide whether the mirrored mechanistic battery is worth running.
- [`CLAIMS.md`](CLAIMS.md) — the claim ledger (`AC*`). Separate from the donor study's `C*` ledger on
  purpose: acceptor reconstruction is a different behaviour from donor reconstruction, and giving it
  its own ledger keeps C1–C42 from silently acquiring reverse-direction rows.
- [`DECISIONS.md`](DECISIONS.md) — chronology, including both pilots and their numbers.
- `scripts/splicebert_acceptor_coupling/` — runners. `src/gfmcd/splicing/reverse_coupling.py` — the
  reusable readout. `results/splicebert_acceptor_coupling/` — artifacts, each with a manifest.

## Scope, standing

One checkpoint (`SpliceBERT.1024nt`), canonical internal short introns, both strands, validation split.
Model-internal throughout — **no external biological assay enters this study at any point.** Nothing
here is a claim about splicing, spliceosome biochemistry, or transcript abundance. Evidence never
promotes across levels; read [`../../docs/EVIDENCE_LADDER.md`](../../docs/EVIDENCE_LADDER.md) and
[`../../docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md) before writing prose here.
