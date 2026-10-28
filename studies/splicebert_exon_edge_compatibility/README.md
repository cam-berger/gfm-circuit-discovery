# SpliceBERT exon-edge compatibility (archive record)

This directory is not a study and holds no claim ledger. It keeps the frozen protocols behind two live
claims, C43 and C44, which live in the [donor
ledger](../splicebert_partner_selection/CLAIMS.md#13-coding-boundary-phase-in-partner-selection-consolidated-from-the-archived-exon-edge-arc).
[`CORRECTIONS.md`](../splicebert_partner_selection/CORRECTIONS.md) R45 governs what may be said about the
arc. Read it before citing anything from the arc; it also records three overstatements in the original
prose that must not be reintroduced from Git history.

## Purpose and results

The arc asked whether SpliceBERT's donor–acceptor coupling tracks the annotated codon reading-frame phase
(0/1/2) of a joined coding exon boundary, a different object from the donor ledger's period-3 carrier.
Consolidation kept two of its eleven claim rows; R45 gives the reason each of the other nine was dropped.

| Was | Now | What it says |
|---|---|---|
| `E7r` | `C43` | Masked-donor direct-partner coupling preferentially favours annotated donor–acceptor coding-boundary phase matches, replicated outcome-blind on 384 fresh blocks, inside a frozen chimeric crossed lattice at one fixed span. It does not license the word "only". |
| `E8` | `C44` | Whether the period-3 carrier implements that preference is unresolved. Four authorization gates failed on the development preflight; the 384-block target was never exposed. `INSTRUMENT_UNRESOLVED` licenses no positive, null, or inverted conclusion. It is not a bounded null, nothing is closed, and it is not a biological no-go. |

## Status

Archived. The rest of the arc is at commit `fd26cc1b` on branch `codex/intron-exon-phase`, in the owner's
local clone only; it was never pushed (R49). Retrieve from there with `git show fd26cc1b:<path>`, and do
not restore it to the active tree. Since R49, C43's target family re-executes from a clean checkout using
the carried raw rows. The auditor itself, and everything for C44, are still not re-executable; R49 lists
what remains attestation only.

## Files

- `preregistrations/`: the frozen protocols behind C43
  (`PREREGISTRATION_FRESH_CROSSED_DIRECT_EDGE_REPLICATION.md`) and C44
  (`PREREGISTRATION_DIRECT_EDGE_P3_RELATIONAL_MEDIATION.md`, Amendments `P3-A1`–`P3-A3`), kept at their
  original paths. The P3 protocol and its amendments are path- and sha256-bound by C44's preflight
  manifests; C43's protocol is bound only by an uncarried file at `fd26cc1b` and is kept so the ten-gate
  claim can be checked. Do not edit them, even additively. They, and the carried
  artifacts' `scope` fields, still say "authentic-module", a retired phrase; read it as chimeric
  ([`docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md)).
- [`results/splicebert_exon_edge_compatibility/`](../../results/splicebert_exon_edge_compatibility/): 18
  tracked files. Twelve are the six controlling artifacts carried at consolidation and their six
  manifests, each pinning its archived runner. R49 added C43's raw cell and native-source rows, and R56 the replication
  cohort, each as a gzip with an archive manifest.
- `scripts/splicebert_exon_edge_compatibility/`: the R49 reproduction script
  (`reproduce_crossed_direct_edge_replication.py`) and the packer that writes the gzip archives.
