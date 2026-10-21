# SpliceBERT donor circuit and positional carriers

This study examines how
SpliceBERT's masked-donor prediction depends on its paired acceptor, how the information is generated and
routed, and where the evidence stops.

Start with the [mechanism overview](../../MECHANISM.md), then consult the [claim ledger](CLAIMS.md) for
each result's evidence level, controlling artifacts, and scope.

## Current result

> A structured causal core within distributed computation under the tested interventions combines a
> predominantly relative positional address with an acceptor-local payload and delivers the resulting
> information to the donor readout.

The donor mechanism is measured on the final multi-species `SpliceBERT.1024nt` checkpoint, using
canonical short introns and a native masked-token logit margin. L3H1 combines attention address and
acceptor value payload, writes a donor-local state, and feeds downstream transport heads.

Clean outputs from the nine non-embedding spine nodes recover coupling when injected, while isolating
the full hand-built component set preserves little standalone faithfulness. The held-out static-ranking
curve places the identified mechanism at the top but has no sparse knee (C27/C33). Neither minimality nor
an autonomous circuit is established. The acceptor-local payload is not explained by the tested bulk
composition covariates; its semantic identity remains unresolved, and “splice-site strength” is withdrawn
(C30/R22).

## Authoritative records

| Record | Purpose |
|---|---|
| [`CLAIMS.md`](CLAIMS.md) | Live claims, evidence levels, controlling artifacts, and scope. |
| [`CORRECTIONS.md`](CORRECTIONS.md) | Full historical ledger and permanent R1–R49 correction record. R1–R24 are table rows; R25–R49 are `##` sections. |
| [`FINDINGS.md`](FINDINGS.md) | Concise scientific narrative and limitations. |
| [`P3_CLOSURE_CONFIRMATORY_REPORT.md`](P3_CLOSURE_CONFIRMATORY_REPORT.md) | C36 design and result; read with the ledger's provenance and sensitivity bounds. |
| [`DECISIONS_A0.md`](DECISIONS_A0.md) | Detailed A0→A12 rationale, amendments, and post-outcome adjudication. |
| [`DECISIONS.md`](DECISIONS.md) | Earlier high-level chronology of methodological pivots. |
| [`METHODS.md`](METHODS.md) | Study-specific discovery, preflight, intervention, and stopping rules. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Detailed script/result map; repository setup is in [`../../REPRODUCE.md`](../../REPRODUCE.md). |
| [`preregistrations/`](preregistrations/README.md) | Indexed frozen, completed, parked, superseded, and deferred protocols. |
| [`redteam/`](redteam/README.md) | P3 adversarial design reviews and execution synthesis. |
| [`archive/`](archive/README.md) | Historical records preserved but not current entry points. |

The claim and correction ledgers control the scientific record; narrative summaries do not supersede
them. [Preregistration conformance](../../docs/PREREG_CONFORMANCE.md) records departures between frozen
protocols and executed code.

## Evidence boundaries

- Donor coupling, the structured core, `U_info`, address/payload mediation, and period-3 carrier use
  have causal support within their stated interventions and data splits.
- C34's confirm split is gate-qualified (R24). C36 provides nucleotide-invariant evidence that the
  learned P3 direction is used in the tested CDS-versus-upstream-intron contrast; coding specificity
  remains unresolved. Required strand and donor-phase sensitivities were not computed, and the current
  source does not match one frozen code pin (R46/R48).
- IC1 is outcome-aware deterministic verification, not confirmation.
- The distinct ≈10.63 carrier has controlled post-softmax attention evidence. Its donor-coupling value
  effects are below the declared meaningfulness threshold, and deletion produces a small suppressive
  effect with underpowered equivalence. The general-MLM-output read is a bounded diagnostic; the
  `U_info` write endpoint remains unresolved and unevaluable as scoped because no L0 reader qualified
  (C12/C37, R31/R36).
- Cross-example donor-state injection transfers a gene-generic readout axis; the example-specific
  component is inert for the donor margin in the tested linear regime (C41, verification tier). This
  bounds additive injection of the final-residual state, not earlier addressing or routing.
- Cross-example acceptor-payload injection moves the donor margin, but its content-specific contribution
  is unresolved at achieved validation precision (C38, verification tier). Payload semantic identity
  remains closed-bounded under C30; transfer alone does not identify it.
- No result here is a direct claim about spliceosome biochemistry or transcript abundance.

## Data and artifacts

- Donor graph: [`../../results/splicebert_coupling/`](../../results/splicebert_coupling/)
- Position/carrier index: [`../../results/phase-lattice/RESULTS_INDEX.md`](../../results/phase-lattice/RESULTS_INDEX.md)
- Active runners: [`../../scripts/splicebert/`](../../scripts/splicebert/)
- Current paper figures: [`headline.png`](../../figures/headline.png) and [`fig_3.png`](../../figures/fig_3.png)

Attribution, probes, attention, and spectra nominate candidates. Interventions establish causal use at
the tested evidence level.
