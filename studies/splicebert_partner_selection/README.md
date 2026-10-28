# SpliceBERT donor circuit and positional carriers

This study asks how SpliceBERT's masked-donor prediction depends on the donor's paired acceptor, how that
information is generated and routed, and where the evidence stops. We mask the donor on the
`SpliceBERT.1024nt` checkpoint, corrupt its paired acceptor in canonical short introns, and read the
native masked-token logit margin. Attribution, probes, attention, and spectra nominate candidate
components; interventions decide causal use at the tested evidence level. The frozen mechanism statement
and its limits are in [`docs/MECHANISM.md`](../../docs/MECHANISM.md), and the [claim ledger](CLAIMS.md)
gives every result's evidence level and scope.

## Findings

Corrupting the paired acceptor changes the masked-donor margin by +0.872 logit units [0.771, 0.969],
while matched non-partner and other-donor controls change it by only +0.007/+0.005 (C1, C2). L3H1
combines a predominantly relative positional address with an acceptor-local value payload, and its effect
routes through a donor-side low-rank direction, `U_info`, to downstream transporter heads (C16–C18, C23,
C32). Injecting clean outputs of the identified nodes recovers coupling, but the hand-built component
set alone has little standalone faithfulness, and the held-out static-ranking faithfulness curve has no
sparse knee. The mechanism is a structured causal core within distributed computation under the tested
interventions; neither minimality nor an autonomous circuit is established (C27, C33). The acceptor-local payload is not
explained by the tested bulk composition covariates; its semantic identity remains unresolved, and the
label "splice-site strength" is withdrawn (C30, R22). A period-3 positional carrier contributes across
the donor–acceptor span through L3H1's attention pattern, and its coding specificity remains unresolved
(C34, C35, C36). C34's confirm split is gate-qualified (R24).

## Status

The A0 to A12 dissection is complete. Later results on cross-example transfer (C38, C41) and the separate
≈10.63 carrier (C12, C37) are recorded with their limits in the ledger, and follow-up tracks are in the
[roadmap](../../docs/ROADMAP.md). No result here is a direct claim about spliceosome biochemistry or
transcript abundance.

## Files

- [`CLAIMS.md`](CLAIMS.md) and [`CORRECTIONS.md`](CORRECTIONS.md) (R1–R59): claims and corrections. They
  control the record; narrative summaries do not supersede them.
- [`FINDINGS.md`](FINDINGS.md): narrative. [`METHODS.md`](METHODS.md): study rules.
  [`DECISIONS_A0.md`](DECISIONS_A0.md) and [`DECISIONS.md`](DECISIONS.md): design rationale.
  [`P3_CLOSURE_CONFIRMATORY_REPORT.md`](P3_CLOSURE_CONFIRMATORY_REPORT.md): the C36 report.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): script and result map; setup is in
  [`docs/REPRODUCE.md`](../../docs/REPRODUCE.md).
- [`preregistrations/`](preregistrations/README.md), [`redteam/`](redteam/README.md),
  [`archive/`](archive/README.md): protocols, design reviews, and historical records. Check
  [preregistration conformance](../../docs/PREREG_CONFORMANCE.md) before citing a gate as enforced.
- Artifacts in [`results/splicebert_coupling/`](../../results/splicebert_coupling/) and
  [`results/phase-lattice/`](../../results/phase-lattice/RESULTS_INDEX.md); runners in
  [`scripts/splicebert/`](../../scripts/splicebert/); figures
  [`headline.png`](../../figures/headline.png) and [`fig_3.png`](../../figures/fig_3.png).
