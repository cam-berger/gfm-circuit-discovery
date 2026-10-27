# SpliceBERT exon-edge compatibility — archive record

> This directory is **not a study**. It holds the two frozen protocols that
> govern two live claims, and this pointer. Nothing here is a claim ledger.

## What this was

An arc asking whether SpliceBERT's donor–acceptor coupling tracks the **annotated codon reading-frame phase**
of a joined coding boundary. It produced **eleven** claim rows — `E0`, `E1`, `E1a`, `E2`, `E3`, `E4`, `E5`, `E6`, `E7`, `E7r`, `E8`
(so `E0`–`E8` is **not** exhaustive).

It was **consolidated, not merged.** Two rows survive, in the donor-circuit ledger:

| Was | Now | What it says |
|---|---|---|
| `E7r` | [`C43`](../splicebert_partner_selection/CLAIMS.md#13-coding-boundary-phase-in-partner-selection-consolidated-from-the-archived-exon-edge-arc) | Masked-donor direct-partner coupling preferentially favours **annotated** donor–acceptor coding-boundary phase matches, replicated outcome-blind on 384 fresh blocks — inside a frozen **chimeric** crossed lattice at one fixed span, and **not** licensing the word "only". |
| `E8` | [`C44`](../splicebert_partner_selection/CLAIMS.md#13-coding-boundary-phase-in-partner-selection-consolidated-from-the-archived-exon-edge-arc) | Whether the period-3 carrier implements that preference is **unresolved**: four authorization gates failed on the development preflight and the 384-block target was never exposed. `INSTRUMENT_UNRESOLVED` licenses **no** positive, null, or inverted conclusion — it is not a bounded null, nothing is closed, and it is not a biological no-go. |

The nine dropped rows are reproduced, with the reason for each, in
[`CORRECTIONS.md` **R45**](../splicebert_partner_selection/CORRECTIONS.md). Read R45 before citing
anything from this arc: it also records three overstatements in the original study's prose that were
corrected during consolidation and must not be reintroduced from Git history.

## Why "phase" here is not §7's period-3 phase

*Phase* in `C43`/`C44` means the annotated **codon reading-frame phase** of a coding exon boundary
(0/1/2). The period-3 carrier of the donor ledger's §7 is a *positional* routing coordinate. `C44` is
exactly the test of whether the second implements the first — and it did not resolve.

## Where the rest is

Everything not carried — the study's own `CLAIMS.md`/`RESULTS.md`/`DECISIONS.md`, 30 runners, 17
`src/gfmcd/splicing/` modules, 30 test files, three further preregistrations with ten amendments, and a
99.4 MB results tree including the 48 MB and 37 MB crossed cohorts — is at:

```
commit fd26cc1b   branch codex/intron-exon-phase   (owner's local clone only; never pushed)
```

On that clone, retrieve with `git show fd26cc1b:<path>`. Do **not** restore it to the active tree. An
earlier version of this README, and R45, said the branch was pushed to `origin`; it was not
([`CORRECTIONS.md` **R49**](../splicebert_partner_selection/CORRECTIONS.md)).

## What is here, and why

- `preregistrations/PREREGISTRATION_FRESH_CROSSED_DIRECT_EDGE_REPLICATION.md` — the frozen protocol
  behind `C43`. Kept because "all ten target-independent gates passed" is only checkable against it.
- `preregistrations/PREREGISTRATION_DIRECT_EDGE_P3_RELATIONAL_MEDIATION.md` + Amendments
  `P3-A1`–`P3-A3` — the frozen protocol behind `C44`, including the decision table whose
  `P3_CARRIER_BOUNDED_NULL` row is the one a reader must consult before concluding anything from
  `C44`'s `Ω`: that rule is defined on the **384-block target** `Ω`, which was never measured, so the
  development-preflight `Ω` must not be compared against its ±0.05 band.

These sit at their **original relative paths** so the path-plus-sha256 bindings that reference them still
resolve. Be precise about which: the bindings live in C44's `development_preflight_analysis.manifest.json`
and `development_preflight_audit.manifest.json` and cover the **P3 protocol and its three amendments**
(path, sha256 and size verified). The two frozen-package audits reference no preregistration at all, and
C43's protocol is bound only by the *uncarried* `crossed_direct_edge_replication_frozen_package.json` at
`fd26cc1b` — it is kept here because the ten-gate claim is not checkable without it. They are frozen: do
not edit them, including additively.

Note that these frozen documents, and the `scope` fields inside the carried artifacts, still contain the
retired phrase **"authentic-module"**. That is correct — immutable records are not rewritten — but read
every such occurrence as **chimeric**; see `docs/NOMENCLATURE.md`.

The controlling artifacts are under
[`results/splicebert_exon_edge_compatibility/`](../../results/splicebert_exon_edge_compatibility/) —
12 files, 4.0 MB. Their `script` column in `results/MANIFEST.csv` is blank by design; each sibling
`*.manifest.json` pins its archived runner by path and sha256. R49 added four more: C43's raw cell rows
as a 4.1 MB gzip, its native-source rows as a 0.2 MB gzip, and the two archives' manifests.

## Known limitation of this archive

Neither carried audit is re-executable from a clean checkout, and this was already true on the source
branch: that branch's own `.gitignore` excludes the 54 MB replication cell rows and the 114 preflight
shards (256 MB) the two audits recompute from. The headline numbers in `C43` and `C44` are re-derivable
from the per-block endpoint tables carried **inside** the artifacts; the remaining gates survive as
recorded attestations. Both ledger rows state this.

**Since R49, C43's target family re-executes from a clean checkout.** The raw rows are carried as
`crossed_direct_edge_replication_cells.json.gz` and `crossed_direct_edge_replication_native.json.gz`,
and

```bash
uv run --extra dev python scripts/splicebert_exon_edge_compatibility/reproduce_crossed_direct_edge_replication.py --out <new path>.json
```

recomputes `Ψ`, `M0`/`M+`/`M−`, the five simultaneous directional bounds, all 99 pseudo-register
controls, the add-one `p`, the native floor calibration, and gates 2–8 from them. It checks every
value against the audit and reproduces it bit for bit. The script transcribes the archived arithmetic
and does not restore it. What it cannot recompute, including the stop-load diagnostics and gates 1, 9
and 10, is listed in R49 and its addendum. The auditor itself, and everything for C44, is still not
re-executable.
