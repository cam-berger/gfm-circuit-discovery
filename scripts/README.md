# Experiment and audit scripts

Use the [reproduction guide](../REPRODUCE.md) for environment and data setup, then follow the owning
study's run map. This index locates entry points; a script's presence does not imply that its target
experiment ran or that a stopped protocol may be resumed.

The [result manifest](../results/MANIFEST.csv) maps recorded artifacts to their producing scripts,
provenance, and figures. [Study claim ledgers](../studies/README.md) control the evidence level and
scientific interpretation.

## Workflow directories

| Directory | Scope | Study and run map |
|---|---|---|
| [splicebert/](splicebert/) | Masked-donor coupling, address/payload interventions, positional carriers, and sibling-checkpoint comparisons | [Donor study](../studies/splicebert_partner_selection/README.md) · [reproducibility map](../studies/splicebert_partner_selection/REPRODUCIBILITY.md) |
| [crossmodel/](crossmodel/) | Positional-carrier comparisons across models and tokenizer controls | [Cross-model artifacts](../results/origin-crossmodel/README.md) · [donor claim ledger](../studies/splicebert_partner_selection/CLAIMS.md) |
| [dnabert_promoter/](dnabert_promoter/) | Frozen-encoder promoter probes and causal interventions | [Promoter study and commands](../studies/dnabert_promoter/README.md) · [artifact index](../results/dnabert_promoter/RESULTS_INDEX.md) |
| [splicebert_acceptor_coupling/](splicebert_acceptor_coupling/) | Retained runners for earlier reciprocal-behavior and component experiments | [Acceptor study](../studies/splicebert_acceptor_coupling/README.md) · [artifact index](../results/splicebert_acceptor_coupling/RESULTS_INDEX.md) |
| [splicebert_head_context_atlas/](splicebert_head_context_atlas/) | Cohort construction, component screens, validation, and conditional-path analyses | [Atlas study](../studies/splicebert_head_context_atlas/README.md) · [artifact index](../results/splicebert_head_context_atlas/RESULTS_INDEX.md) |
| [splicebert_gtex_finetune_shift/](splicebert_gtex_finetune_shift/) | Readout transfer, native-classifier interventions, and positional/representation follow-ups | [Fine-tuning study](../studies/splicebert_gtex_finetune_shift/README.md) · [results and commands](../studies/splicebert_gtex_finetune_shift/RESULTS.md) |
| [splicebert_positional_boundary_coherence/](splicebert_positional_boundary_coherence/) | Construction, freeze, and audit tools for the successor boundary-coherence design; execution stopped at model-free capacity | [Boundary-coherence study and gate chain](../studies/splicebert_positional_boundary_coherence/README.md) |
| [splicebert_exon_edge_compatibility/](splicebert_exon_edge_compatibility/) | Model-free re-execution of C43's crossed-register target family from its archived raw rows (R49); no producing runner | [Archive record](../studies/splicebert_exon_edge_compatibility/README.md) · [R49](../studies/splicebert_partner_selection/CORRECTIONS.md) |
| [splicebert_positional_clock/](splicebert_positional_clock/) | Retained model-free capacity analyses; the historical directory name is preserved | [Positional-proxy study](../studies/splicebert_positional_proxy_modifiability/README.md) · [artifact index](../results/splicebert_positional_clock/RESULTS_INDEX.md) |

## Main donor workflow

These entry points locate the principal assay families. Exact commands, dependencies, controls, and
historical qualifications are in the [study reproduction map](../studies/splicebert_partner_selection/REPRODUCIBILITY.md)
and [preregistration conformance audit](../docs/PREREG_CONFORMANCE.md).

| Scientific question | Entry points |
|---|---|
| Does the masked donor depend on its paired acceptor? | [Sealed behavior assay](splicebert/run_sealed_test.py) |
| What does attention select, and what do values carry? | [Address/payload factorial](splicebert/run_a2_address_payload.py), [source localization](splicebert/run_a7_source_localization.py) |
| How is the acceptor-local payload generated? | [Generator path](splicebert/run_a8_generator_path.py), [generator edges](splicebert/run_a10_generator_edges.py) |
| What is written at the donor, and how is it delivered? | [Donor anchor](splicebert/run_donor_anchor.py), [donor heads](splicebert/run_donor_heads.py), [write replication](splicebert/run_a5_uinfo_replication.py) |
| How much of the computation does the core explain? | [Core faithfulness](splicebert/run_a6_circuit_faithfulness.py), [faithfulness–size curve](splicebert/run_a12_faithfulness_size.py) |
| How is the period-3 positional carrier used? | [Carrier locus](splicebert/run_p3_carrier_locus.py), [attention/value decomposition](splicebert/run_p3_av_decomposition.py), [phase-surgery preflight](splicebert/run_p3_phase_surgery_preflight.py), [phase-surgery target](splicebert/run_p3_phase_surgery_confirmation.py) |
| Does a derived state transfer across examples? | [Acceptor-side transduction](splicebert/run_b3_cross_example_transduction.py), [donor-side transduction](splicebert/run_g1_cross_example_transduction.py) |

The sealed assay is a historical experiment, not a smoke test. Follow its
[frozen protocol](../studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_SEALED.md)
and the reproduction guide's access rules before executing it.

## Repository integrity

Run from the repository root:

```bash
python3 scripts/verify_manifest.py
```

[verify_manifest.py](verify_manifest.py) checks result coverage, referenced paths, status and polarity
vocabularies, and agreement between the CSV and JSON indexes. It checks index integrity; it does not
validate scientific estimates or certify protocol conformance.

[verify_archived_evidence.py](verify_archived_evidence.py) verifies compact attestations against their
archived and active bytes. For example:

```bash
python3 scripts/verify_archived_evidence.py \
  results/splicebert_acceptor_coupling/attestations/paired_donor_component_confirmation_v1.json
```

Archive verification requires the pinned Git objects to be available locally. Consult the relevant
attestation and [reproduction guide](../REPRODUCE.md) for acquisition and replay limits.

## Figure rendering

These renderers read recorded results without running model experiments:

- [SpliceBERT mechanism figure](splicebert/make_mechanism_figure.py) — period-3 use, address/payload
  recovery, faithfulness, and Fourier-plane geometry.
- [DNABERT-2 comparison](dnabert_promoter/make_paper_figure_transfer.py) — probe, component, and
  faithfulness panels.

See the [figure index](../figures/README.md) for source sidecars, output commands, and interpretation
limits. Historical `make_fig_*_data.py` scripts produce analysis artifacts and are not layout renderers.

## Execution and provenance

- Run each study's preflight before its target experiment, with frozen thresholds and stopping rules.
- Write new results to unused paths with machine-readable output, provenance, and the required audit.
  Preserve earlier results, failed controls, and retractions.
- Generate figures from recorded artifacts and declare their sources.
- Keep study-specific runners in their owning directory. Reusable implementation belongs under
  [src/gfmcd/](../src/gfmcd/).

Historical manifests preserve the command as executed. For example, a GTEx manifest may name
`scripts/splicebert/run_gtex_*.py`, while the corresponding runner now lives under
`scripts/splicebert_gtex_finetune_shift/`. Current run maps resolve moved code without rewriting that
provenance.

The [exon-edge archive record](../studies/splicebert_exon_edge_compatibility/README.md) explains why
C43/C44 have no producing runners here. Their archived code remains archived; the successor study
does not import it, and the failed C44 authorization path remains closed. The
`splicebert_exon_edge_compatibility/` directory holds only R49's raw-row packer and C43 reproduction
script. The script transcribes the archived arithmetic and does not restore it.
