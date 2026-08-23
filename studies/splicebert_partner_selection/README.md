# SpliceBERT donor circuit and positional carriers

This directory is the detailed provenance record for the repository's first worked example: how
SpliceBERT's masked-donor prediction depends on its paired acceptor, how the information is generated and
routed, and where the evidence stops.

For the shortest reviewer path, read the repository-level [`../../MECHANISM.md`](../../MECHANISM.md),
then the compact [`CLAIMS.md`](CLAIMS.md).

## Current result

> A structured, top-ranked causal core combines a positional address with an acceptor-local payload, then
> delivers the resulting information to the donor readout within distributed computation under the
> tested interventions.

The named core is causally necessary and its clean outputs are sufficient when injected, but it is not a
minimal or self-contained generator. The held-out faithfulness curve has no sparse knee. The primary
semantic limitation is that the payload remains unnamed: it is acceptor-local and not bulk composition,
but “splice-site strength” was withdrawn by R22.

## Authoritative records

| Record | Purpose |
|---|---|
| [`CLAIMS.md`](CLAIMS.md) | Compact live claims, evidence level, controlling artifact, and scope. |
| [`CORRECTIONS.md`](CORRECTIONS.md) | Full historical ledger and permanent R1–R48 correction record. R1–R24 are table rows; R25–R48 are `##` sections. |
| [`FINDINGS.md`](FINDINGS.md) | Concise scientific narrative and limitations. |
| [`P3_CLOSURE_CONFIRMATORY_REPORT.md`](P3_CLOSURE_CONFIRMATORY_REPORT.md) | Clean C36 design, result, provenance, and licensed claim. |
| [`DECISIONS_A0.md`](DECISIONS_A0.md) | Detailed A0→A12 rationale, amendments, and post-outcome adjudication. |
| [`DECISIONS.md`](DECISIONS.md) | Earlier high-level chronology of methodological pivots. |
| [`METHODS.md`](METHODS.md) | Study-specific discovery, preflight, intervention, and stopping rules. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Detailed script/result map; repository setup is in [`../../REPRODUCE.md`](../../REPRODUCE.md). |
| [`preregistrations/`](preregistrations/README.md) | Indexed frozen, completed, parked, superseded, and deferred protocols. |
| [`redteam/`](redteam/README.md) | P3 adversarial design reviews and execution synthesis. |
| [`archive/`](archive/README.md) | Historical records preserved but not current entry points. |

The original C16–C19 donor-circuit synthesis — a T3 historical document — is no longer in the tree. It
predated A0→A12 and R22–R24; current framing is controlled by the repository-level
[`../../MECHANISM.md`](../../MECHANISM.md) and the compact [`CLAIMS.md`](CLAIMS.md) above.

## Evidence boundaries

- Donor coupling, the structured core, `U_info`, address/payload mediation, and period-3 carrier use
  reach behavior-level causal evidence within their stated interventions.
- C34's confirm split is gate-qualified (R24). C36 provides nucleotide-invariant evidence that the
  learned P3 direction is used in the tested CDS-versus-upstream-intron contrast; coding specificity
  remains unresolved, and the C36 code-pin caveat is recorded under R46/R48.
- IC1 is outcome-aware deterministic verification, not confirmation.
- The distinct multispecies ≈10.63 carrier stops at post-softmax attention.
- Cross-example/context transduction and payload semantic identity are open.
- No result here is a direct claim about spliceosome biochemistry or transcript abundance.

## Data and artifacts

- Donor graph: [`../../results/splicebert_coupling/`](../../results/splicebert_coupling/)
- Position/carrier index: [`../../results/phase-lattice/RESULTS_INDEX.md`](../../results/phase-lattice/RESULTS_INDEX.md)
- Active runners: [`../../scripts/splicebert/`](../../scripts/splicebert/)
- Current figures: [`../../figures/splicebert/`](../../figures/splicebert/README.md)

The reading rule is simple: artifacts and corrections bound the prose. Attribution, probes, attention,
and spectra can generate a candidate; only a qualified intervention can promote it.
