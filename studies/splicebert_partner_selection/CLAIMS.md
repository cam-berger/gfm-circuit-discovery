# SpliceBERT live claim ledger

This is the point-of-contact record for the strongest wording currently licensed by the evidence. Each
row is deliberately limited to one claim, one evidence level, one controlling artifact set, and one
scope boundary. Full numerical detail, superseded narratives, and permanent retractions R1–R24 are
preserved in [`CORRECTIONS.md`](CORRECTIONS.md); the experiment chronology is in
[`DECISIONS_A0.md`](DECISIONS_A0.md).

Status vocabulary:

- **established** — supported at the stated evidence level by indexed artifacts and qualified controls;
- **bounded** — supported only within an explicit power, intervention, or scope boundary;
- **open** — the necessary evidence level has not been reached;
- **retracted** — historical wording that must not be presented as live;
- **closed** — a terminal row: no further evidence will be sought (a permanent power/sensitivity bound, or a
  forward-pointing stub superseded by a stronger row); retained for provenance, not a live loose end.

## Live claims

| ID | Current licensed claim | Evidence level | Controlling artifact(s) | Scope / first boundary |
|---|---|---|---|---|
| C1 | Corrupting a donor's paired acceptor changes the native masked-donor margin by **+0.872 [0.771, 0.969]**. | established: behavior | `results/splicebert_coupling/sealed_test.json` | Canonical short introns, sealed chr8/9 split, one checkpoint. |
| C2 | The coupling is partner- and donor-specific: matched non-partner AG and other-donor controls change the margin by only **+0.007/+0.005**. | established: specificity | `sealed_test.json` | Specific to the frozen controls; not a transcript-level splicing assay. |
| C3 | `{L3H1, L2H15}` is a dominant necessary subcircuit: ablation removes **81.6%** of coupling while retaining **99.2%** masked-MLM health. | established: necessity | `sealed_test.json` | Dominant does not mean complete or minimal. |
| C4 | L3H1's acceptor read is conjunctive: key-only and value-only effects are small, while key+value gives **+0.663** with interaction **+0.531 [0.477, 0.583]**. | established: mediation | `sealed_test.json` | Conditional on the sealed patching intervention. |
| C5 | L3H1 reads at the acceptor, writes into the donor ±5 neighborhood, and feeds a distributed downstream readout. | established: localization/path | `position_profile.json`; `readout_path.json` | Component-level path, not feature-level completeness. |
| C6 | L2H15 and L1H12 contribute upstream through L3H1, but neither supports a simple exclusive-role story. | established with negative interpretation | `l2_accounting.json`; `l1h12_route.json` | Marginal effects are not additive shares. |
| C7 | The early broad-ablation completeness assay was uninformative; completeness is instead bounded by C27/C33. | closed: superseded by C27/C33 | `completeness_null.json`; `a6_circuit_faithfulness.json`; `a12_faithfulness_size.json` | Do not cite the damaged-model null as evidence of completeness. |
| C8 | A rank-2 learned period-3 position code causally supports coding prediction and donor–acceptor coupling; deletion changes coupling **0.946 → 0.027**. | established: position through behavior | `carrier_surgery.json`; `carrier_dose.json` | Model mechanism; no biological-origin claim; C36 supplies the clean phase-aligned-use confirmation. |
| C9 | Restoring L2H15 or L3H1 after period-3 deletion recovers **0.363/0.348**, above the matched one-head null maximum **0.042**. | established: size-matched mediation | `carrier_mediation.json` | Single-head mediation under the carrier-deletion intervention. |
| C10 | Six- and sixteen-head restorations do not independently strengthen C9 because their matched-null support is inadequate. | closed: bounded (permanent matched-null power limit) | `carrier_mediation.json` | No larger-set superiority claim. |
| C11 | The distinct multispecies **≈10.63** carrier reaches controlled post-softmax attention in eight heads. | established: routing/attention | `postsoftmax.json`; `preflight_postsoftmax.json`; `band_mass.json`; `fig_report_data.json` | Separate from the period-3 carrier; stops at attention. |
| C12 | The ≈10.63 carrier's OV content, behavioral function, and biological origin remain unknown. | open | absence of higher-level indexed evidence | Do not infer use or biology from its spectral/attention signature. |
| C13 | The proposed human-only ≈10.2 chromatin circuit has zero corrected Level-A heads and is closed at the attention gate. | established: powered attention null | `postsoftmax.json`; `band_mass.json`; `fig_report_data.json` | Refutes the proposed reader set, not every possible chromatin representation. |
| C14 | A real nucleosomal signal occurs near **10.298 nt**, distinguishable from the model's ≈10.63 carrier. | established: biological-source distinction | `dyad_real.json`; `dyad_wavelength.json` | Supports distinctness, not the origin of either carrier. |
| C15 | Unanchored and splice-window assays set sensitivity limits but do not establish absence of weak biological periodicity. | closed: bounded (permanent sensitivity limit) | `splice_profile.json`; `splice_phasing.json`; `splice_long.json` | Nulls are limited by calibrated resolution and power. |
| C16 | A stable low-rank final-residual direction, `U_info`, mediates donor coupling; rank-8 patching recovers about **0.92** versus matched random subspaces ≤**0.05**. | established: output/readout | `donor_anchor.json` | Donor-side delivered representation, not semantic identity. |
| C17 | Restoring L3H1 recovers **0.83** of coupling; **0.93** of that effect routes through `U_info`, predominantly at donor ±5. | established: mediation/path | `donor_forward.json` | Conditional mediation; fractions are not additive across serial components. |
| C18 | L5H11, L4H2, and L4H0 are selective downstream transporters of the donor-side information. | established: head-level mediation | `donor_heads.json` | Transporter set is scoped to the tested checkpoint and behavior. |
| C19 | The delivered signal is partner- and donor-specific, strand-symmetric, and insensitive to intron-length mod 3 or annotated splice phase. | established: behavioral signature | `donor_signature.json` | Describes model behavior; does not name payload semantics. |
| C20 | L3H1 empirically prefers the paired acceptor in QK/attention, but the preference is mostly positional geometry—not evidence of partner identity. | established statistic; identity retracted (R20) | `acceptor_address.json`; `a1_token_position.json` | The separate behavioral specificity in C19 remains intact. |
| C21 | L3H1's acceptor-targeting difference is already present in embeddings; layers 0–2 mainly add class-level acceptor-attraction gain. | established: QK/attention provenance | `a0_targeting_provenance.json` | Does not adjudicate biological identity. |
| C22 | L3H1 targeting is dominated by position-embedding geometry within the four-token key window, with a smaller reproducible content contribution. | established: QK/selection | `a1_token_position.json` | Nonlocal context may have been compressed into the local state earlier. |
| C23 | In a non-forced reconstruct–inject factorial, L3H1's coupling contribution requires the positional attention address and acceptor value payload jointly. | established: behavior-level mediation | `a2_address_payload.json` | Partial L3H1 mechanism; not complete behavior or payload meaning. |
| C24 | The payload is localized to the four-token acceptor window and is not concentrated in the donor-derived rank-8 `U_info`; L3H1 converts representations. | established: payload localization | `a2b_payload_localization.json` | Intrinsic acceptor-payload dimensionality remains unmeasured. |
| C25 | L2H15 acts through L3H1's address channel, amplifying positional acceptor attraction rather than supplying the payload or originating the geometry. | established: behavior-level attribution | `a3_l2h15_address_payload.json` | Address-channel contribution is conditional on the reconstruct–inject design. |
| C26 | Independently derived `U_info` directions generalize across disjoint genes (rank-1 overlap **0.970**; held-out rank-8 sufficiency **+0.875**). | established: replication | `a5_uinfo_replication.json` | Validates the donor-side object without redefining it. |
| C27 | The identified delivery spine is necessary (`N(C without embed)=0.989`, post-hoc) and sufficient when injected (`D(C without embed)=0.986`) but is not a faithful sparse generator (`F(C)=0.005`). | established with correction R21 | `a6_circuit_faithfulness.json`; `a6_reconcile.json` | `N(C)=1` and two whole-cover memorials are architecture-forced; minimality is uninterpretable at F≈0. |
| C28 | Payload generation converges locally at the acceptor and depends on a serially interdependent L0→L1→L2 module. | established: source localization | `a7_source_localization.json` | PPT/branchpoint null is limited to this pre-L3H1 coupling channel. |
| C29 | The acceptor-payload generator uses a held-out-generalizing L0→L1→MLP2 MLP backbone; MLP2 has dual address/payload roles and L2H15 is payload-null. | established: generator path | `a8_generator_path.json` | Value path with address held clean; individual L1/L2 heads unresolved. |
| C30 | The payload is acceptor-local and not explained by preregistered bulk composition, but its semantic identity is unresolved; “splice-site strength” is withdrawn. | bounded (R22) | `a9b_payload_semantics_composition.json`; `a9c_matched_composition_reconcile.json` | MaxEnt shows only a weak, sample-variable, non-dyadic-significant tendency. |
| C31 | The resolved value-path wiring is `{L0H7, L0H12, L0H10} → MLP0 → MLP1 → MLP2`, with partial redundancy between H12/H10. | established: generator wiring | `a10_generator_edges.json` | Conditional causal effects in a serial chain; do not sum them. |
| C32 | L3H1's donor→acceptor address is predominantly relative-offset based, with modest absolute-position modulation. | established: QK/selection geometry | `a11_address_geometry.json` | Edge shifts may include position-table boundary artifacts; no direct behavior claim. |
| C33 | Faithfulness has no sparse knee: about **72/103** nodes are needed for 90%, while the identified mechanism forms the top-ranked structured core. | established: held-out faithfulness curve | `a12_faithfulness_size.json` | Static ranking is an upper bound on the minimal faithful set; no smallest-circuit claim. |
| IC1 | Distal intron arrangement has a small sub-SESOI average effect and no detected partner-specific component (`+0.006 [−0.011,+0.024]`). | outcome-aware deterministic verification | `ic1_distal_intron_necessity_v8.json` | Not confirmatory because the outcome was seen before final deterministic/verdict fixes. |
| C34 | The period-3 carrier's coupling contribution is distributed across the donor–acceptor span; site-local restoration recovers only about 2–4%. | established on discovery; gate-qualified verification on confirm | `p3_carrier_locus.json` | Confirm M1 missed its frozen threshold (5.5% > 5%); not a clean confirmatory replication. |
| C35 | Within carrier patching, coupling recovery is mediated mainly by L3H1's attention pattern (**+0.284** confirm) rather than carrier-dependent changes in its values (**−0.005** confirm). | established: A-vs-V conditional mediation | `p3_av_decomposition.json` | Does not exclude upstream carrier effects on payload construction; `Δ_bind` is architecture-forced (R23/R24). |
| C36 | Nucleotide-invariant local surgery confirms that SpliceBERT causally uses its learned P3 hidden direction in a phase-aligned manner for coupling in CDS blocks: **Ψ_C=+0.141 [0.110,0.172] nats**, n=943; the learned direction exceeds all 19 same-geometry controls, with paired excess **+0.136 [0.051,0.222]**. | established: clean confirmatory behavior + direction specificity | `p3_phase_surgery_confirmation_v1/confirmation_analysis.json`; frozen package/audit | Present-day checkpoint use only. The matched CDS-minus-non-CDS-exon contrast is **+0.006 [−0.080,+0.093]**, inconclusive; coding enrichment and training/biological origin are not established. |

## Interpretation guardrails

- A **circuit** here is a causally supported collection of components responsible for a scoped model
  behavior. Unless explicitly stated, it is not claimed to be the smallest such collection.
- C27 and C33 specifically rule out a sparse-faithful reading at the current node granularity: the named
  circuit is a structured causal core embedded in a broadly distributed computation.
- The key scientific limitation is C30: the payload is acceptor-local and not bulk composition, but what
  biological/model variable it represents remains open.
- C36 closes the clean period-3 **current-use** question, not its origin: the model uses its learned P3
  direction in CDS blocks, but the matched non-CDS-exon boundary does not establish coding enrichment.
- **Period-3 present-day-use family (C34/C35/C36) — DONE (2026-07-22).** C36's clean nucleotide-invariant
  confirmation (`Ψ_C=+0.141 [0.110,0.172]`, learned direction above all 19 same-geometry controls) supplies
  the clean phase-aligned-use evidence that C34's gate-qualified confirm (R24) lacked; C35 fixes the routing
  locus (L3H1 attention, not carrier-dependent values). Present-day causal **use** is established; only the
  training/biological **origin** remains, and it is deferred (P3.4 needs training-time evidence — see
  `CLAUDE.md`). No further present-day-use experiment is licensed on this family.
- **Cross-study positive control (donor circuit):** the Head–Context Atlas competition branch (now closed)
  independently rediscovered **L3H1** as the donor-reconstruction mediator on a disjoint donor-target cohort
  (HCA9: donor B +0.114 [0.098,0.130]; exact 400-gene L3H1 patch +0.0857 [0.0492,0.1280] vs behavioral +0.1015),
  corroborating C3/C4/C5/C17. See `../splicebert_head_context_atlas/CLAIMS.md` (HCA9, closed) and its
  `DECISIONS.md` 2026-07-22.
- Historical tripwires remain active: **D30 is a SENSITIVITY LIMIT, not a null**, and the small-model D29
  result does not generalize—**Do NOT write 'transformers cannot learn relative rulers.'**
- For the exact correction history and superseded wording, consult [`CORRECTIONS.md`](CORRECTIONS.md)
  before quoting any claim.
