# SpliceBERT claim and retraction ledger

This ledger controls current wording. It intentionally contains fewer claims than the historical registry: duplicate, superseded, and exploratory rows have been consolidated. Removed detail remains in Git history and indexed artifacts.

Status vocabulary:

- **established** — supported at the named evidence level by an indexed artifact and qualified controls;
- **bounded** — a positive or null with an explicit scope, power, or resolution boundary;
- **open** — not yet tested at the required evidence level;
- **retracted** — must not be presented as a live result.

## Live claims

| ID | Claim | Status | Controlling evidence |
|---|---|---|---|
| C1 | Corrupting the paired acceptor changes the native donor score on short, canonical introns. Sealed effect **+0.872 [0.771, 0.969]**, n = 400. | established: behavior | `results/splicebert_coupling/sealed_test.json` |
| C2 | The effect is partner-specific: non-partner AG **+0.007** and distance-matched other donor **+0.005** on the sealed split. | established: specificity | `sealed_test.json` |
| C3 | `C = {L3H1, L2H15}` is a dominant necessary subcircuit: **81.6%** coupling removed, **99.2%** masked-MLM health retained, matched two-head **p = 0.0025**. | established: necessity | `sealed_test.json` |
| C4 | L3H1's acceptor read is conjunctive: key **0.032**, value **0.101**, key+value **0.663**, interaction **+0.531 [0.477, 0.583]**. | established: mediation | `sealed_test.json` |
| C5 | L3H1 reads at the acceptor, writes at the donor ±5 neighborhood, and is followed by a distributed downstream chain. | established: localization/path | `position_profile.json`, `readout_path.json` |
| C6 | L2H15 and L1H12 contribute upstream through L3H1; neither licenses a simple “address versus query” story. | established with negative interpretation | `l2_accounting.json`, `l1h12_route.json` |
| C7 | Circuit completeness is unquantified. Large-complement ablations are confounded by general model damage and do not distinguish C-preserving sets from size-matched nulls. | bounded/inconclusive | `completeness_null.json` |
| C8 | A rank-2 learned period-3 position code supports coding prediction and donor–acceptor coupling. Deletion changes coupling **0.946 → 0.027**. | established: position through behavior | `carrier_surgery.json`, `carrier_dose.json` |
| C9 | Restoring L2H15 or L3H1 in the period-3-deleted model recovers **0.363/0.348**, above the one-head null maximum **0.042**; exact add-one **p = 0.040**. | established: size-matched mediation | `carrier_mediation.json` |
| C10 | The six-head and 16-head restorations do not independently strengthen C9: the former has large size-matched nulls and the latter has none. | bounded | `carrier_mediation.json` |
| C11 | The multispecies position table contains a separate ≈**10.630** carrier read by eight heads after softmax. | established: Level A attention | `postsoftmax.json`, `preflight_postsoftmax.json`, `band_mass.json`, `fig_report_data.json` |
| C12 | For the multispecies carrier, OV content, behavioral function, and biological origin are unknown. | open | absence of higher-level indexed evidence |
| C13 | The human-only ≈10.2 feature has no Level-A head after exact add-one permutation correction; the proposed chromatin circuit stops at QK. | established: powered attention null | `postsoftmax.json`, `band_mass.json`, `fig_report_data.json` |
| C14 | A dyad-anchored nucleosomal sequence signal exists at ≈**10.298**, not ≈10.630, and does not explain the multispecies carrier. | established: biological source distinction | `dyad_real.json`, `dyad_wavelength.json` |
| C15 | Local splice-junction phasing exists, but the short support cannot resolve 10.298 from 10.630; the long-intron arm is a powered null and the long-exon arm is underpowered. | bounded | `splice_profile.json`, `splice_phasing.json`, `splice_long.json` |
| C16 | The masked-donor coupling is mediated by a low-rank, specific direction in the **final residual** (the "information object" `U_info`). Patching only the top-8 clean−corrupt directions into the corrupt run recovers **0.92** of the coupling (top-1 **0.74**) through the exact MLM head, versus **≤0.05** for size-matched random rank-k subspaces (add-one **p = 0.005**); necessity mirrors it. Injecting a no-donor-context baseline collapses the **+4.65**-nat donor margin to the base-rate floor (R8 alive-check). Anchors the output end of the L3H1→donor-logit trace. | established: output readout, size-matched null | `donor_anchor.json`, `PREREGISTRATION_DONOR_ANCHOR.md` |
| C17 | **L3H1 delivers `U_info`.** Restoring L3H1's clean output into the corrupt run recovers **0.83** of the coupling (matching `readout_path`), and **0.93** of that routes through `U_info` (patching only its rank-8 component; random rank-8 **≤0.05**, add-one **p = 0.005**). The delivery is localized to the donor **±5** neighborhood (**0.96** of the routing; elsewhere **0.06**); the restoration direction overlaps `U_info` at **0.905**. The acceptor-dependent `U_info` write is assembled at **MLP3 and MLP4** (MLP5 writes `U_info` acceptor-independently). Joins the forward and backward arms of the L3H1→donor-logit trace. A residual **7%** of L3H1's recovered margin routes outside rank-8 `U_info` (open). The layer-4/5 attention heads are not yet resolved (1c). | established: mediation/path, size-matched null | `donor_forward.json`, `PREREGISTRATION_DONOR_FORWARD.md` |

## Permanent retractions and corrections

| ID | Withdrawn statement or practice | Why it failed | Replacement rule |
|---|---|---|---|
| R1 | A randomized-backbone control supported a v1 partner-selection claim. | The randomization call was a no-op; weights remained pretrained. | Instantiate random controls from config and verify parameter differences numerically. |
| R2–R3 | The v1/v2 forced-choice task measured splice-partner selection. | Fixed positions leaked labels; site detection sufficed; “real” decoys were weaker minor sites. | Use native cross-site coupling with matched motif and distance controls. |
| R4–R6 | L3H1 was a single dedicated head, its marginal effect was a share of total causal mass, and L2H15 refined key and value. | Held-out factorials found a local role; marginal effects double-count serial components; causal rescue contradicted cosine similarity. | Say dominant subcircuit; use factorial/path interventions; do not convert attribution or cosine magnitude into causal shares. |
| R7 | Key/value swapping demonstrated separable address and content. | Swapping complete `(k,v)` pairs permutes a sum and is algebraically invariant. | Reject interventions whose answer is forced by architecture; use manipulations that can fail. |
| R8 | Near-perfect unmasked-token accuracy showed preserved model health. | The model copied visible inputs; the control could not fail. | Mask health probes and demonstrate collapse under a destructive intervention. |
| R9 | A duplicated random null gave **p = 0.000999**. | Sampling with replacement duplicated 95 deterministic units and made p a function of requested draws. | Enumerate/de-duplicate finite nulls and use exact add-one p-values. |
| R10 | Held-out coupling was **1.359**. | The mean was conditioned on coupling ≥0.3. | Report unconditional behavior; reserve conditioned sets for recovery ratios and label them. |
| R11 | Early results covered both strands. | A transcript-order bug silently dropped every minus-strand exon. | Validate eligible counts by strand; the corrected circuit is unchanged. |
| R12–R13 | Broad complement ablation proved a downstream stage; a destroyed model provided a functional null for period 3. | Size-matched nulls were equally destructive and the model fell to chance. | Match intervention size/health and never interpret a mechanism after destroying the measured behavior. |
| R14 | Forty-two heads formed a codon-phase attention lattice. | Phase purity rewarded one-offset attention and measured sharpness. | Require repeated comb teeth plus deletion and protected controls. |
| R15 | Relocating key/value vectors demonstrated content-addressed, position-independent lookup. | Absolute position was already inside the vectors; the result was architecturally forced. | Do not claim a pointer or content-addressed mechanism. The valid positional result is C8. |
| R16 | A weak helical estimator refuted a chromatin interpretation. | Its measured detection floor exceeded the expected signal. | Compute sensitivity and resolution before interpreting a null. |
| R17 | The multispecies ≈10.63 carrier was the ≈10.3 nucleosomal signal. | A powered anchored assay found a coherent biological signal at the wrong wavelength. | Treat the multispecies carrier's origin as open. |
| R18 | An “inert” band controlled the human-only experiment and bounded a nucleosomal fraction. | The band was strongly read; bare-token inputs were off format; the corrected effect changed sign and lost power. | Name controls only by measured properties and verify tokenizer/input format. |
| R19 | A raw zero-hit permutation value could be reported as **p = 0** and survive family correction. | Zero hits in 400 permutations is a bound, not a zero p-value. | Use `(hits + 1)/(n + 1)` before multiple-testing correction; human-only Level A becomes zero heads. |

## Interpretive firewall

For every candidate, record the highest level reached and do not auto-promote:

| Level | Required evidence |
|---|---|
| P | a feature exists in position or representation space |
| QK | queries/keys read the feature in logits |
| A | the signal survives softmax into attention with qualified controls |
| OV | head output carries relevant information or mediates a path |
| Behavior | a targeted intervention changes a model-native behavior while health controls pass |
| Biology | an independently measured biological source or function matches with adequate power and specificity |

The multispecies ≈10.63 carrier stops at A. The human-only ≈10.2 proposal stops before A. The donor–acceptor/period-3 result reaches behavior. None of these automatically establishes a biological mechanism.
