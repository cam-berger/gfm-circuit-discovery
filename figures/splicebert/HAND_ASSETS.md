# Hand-authored SpliceBERT assets — retirement record

This file exists to keep a retraction attached to artwork that no longer sits in the tree. It is not a
figure index — the figures of record and their generating scripts are in [`README.md`](README.md).

## Removed from the active tree (2026-08-10)

None of these were produced by the figure pipeline: they carried no `.provenance.json`, named no source
artifact, and nothing in the repository referenced them. They were retained as historical artwork rather
than scientific inputs, and one of them asserts a retracted interpretation (below). Git history retains
all four.

- `figures/splicebert/splicebert_donor_acceptor_binding.svg`
- `figures/splicebert/splicebert_donor_acceptor_binding_preview.png`
- `figures/splicebert/splicebert_donor_acceptor_binding_grayscale.png`
- `studies/splicebert_partner_selection/circuit.svg`

Superseded phase snapshots and the stale `fig4_carrier_circuit` asset were removed earlier under the same
rule.

## Why the binding artwork must not be recovered and cited

The donor–acceptor binding illustration asserts a **content-addressed, position-independent read**. That
interpretation is **retracted as R15**: moving a key/value pair that already contains absolute position
information makes the apparent relocation result architecturally forced. Recovering the file from Git
history and putting it in a talk or a paper would reintroduce a claim the project has withdrawn.

The current mechanism is in
[`../../studies/splicebert_partner_selection/FINDINGS.md`](../../studies/splicebert_partner_selection/FINDINGS.md)
and [`../../MECHANISM.md`](../../MECHANISM.md). Both support a conjunctive acceptor read and a period-3
positional contribution, and neither licenses a pointer or content-addressed-lookup claim.

## Canonical publication set

| Figure | Generator |
|---|---|
| `fig1_summary` | `scripts/splicebert/make_fig1_summary.py` |
| `fig2_oscillators` | `scripts/splicebert/make_fig2_oscillators.py` |
| `fig3_clock_circuit` | `scripts/splicebert/make_fig3_clock_circuit.py` |
| `fig5_p3_phase_surgery_confirmation` | `scripts/splicebert/make_fig5_p3_phase_surgery_confirmation.py` |

The machine-readable results, not the artwork, control exact values.
