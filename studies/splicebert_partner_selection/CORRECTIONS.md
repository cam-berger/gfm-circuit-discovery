# SpliceBERT corrections and full historical claim record

> **Tier: T1 — canonical.** Current, authoritative. Read this.

This file preserves the complete pre-condensation claim ledger verbatim, including superseded narratives,
R-number corrections, calibration detail, and the permanent retraction table. It is the provenance record
for how interpretations changed; it is intentionally dense.

For the current licensed wording, use [`CLAIMS.md`](CLAIMS.md). If a compact claim conflicts with a
correction here, the correction and its controlling artifact set the boundary. Nothing was deleted when
the point-of-contact ledger was condensed on 2026-07-20.

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
| C7 | Circuit completeness is unquantified. Large-complement ablations are confounded by general model damage and do not distinguish C-preserving sets from size-matched nulls. **[Quantified → A6/C27 (2026-07-16): scrub-**to-corrupt** (not mean-ablation; exact healthy memorials) gives faithfulness `F(C)=0.005`, damage confound excluded — the necessary circuit is 99.5% non-faithful because *generation* is distributed; its clean outputs remain sufficient (`D=0.986`).]** | bounded → quantified (C27) | `completeness_null.json`, `a6_circuit_faithfulness.json` |
| C8 | A rank-2 learned period-3 position code supports coding prediction and donor–acceptor coupling. Deletion changes coupling **0.946 → 0.027**. | established: position through behavior | `carrier_surgery.json`, `carrier_dose.json` |
| C9 | Restoring L2H15 or L3H1 in the period-3-deleted model recovers **0.363/0.348**, above the one-head null maximum **0.042**; exact add-one **p = 0.040**. | established: size-matched mediation | `carrier_mediation.json` |
| C10 | The six-head and 16-head restorations do not independently strengthen C9: the former has large size-matched nulls and the latter has none. | bounded | `carrier_mediation.json` |
| C11 | The multispecies position table contains a separate ≈**10.630** carrier read by eight heads after softmax. | established: Level A attention | `postsoftmax.json`, `preflight_postsoftmax.json`, `band_mass.json`, `fig_report_data.json` |
| C12 | For the multispecies carrier, OV content, behavioral function, and biological origin are unknown. | open | absence of higher-level indexed evidence |
| C13 | The human-only ≈10.2 feature has no Level-A head after exact add-one permutation correction; the proposed chromatin circuit stops at QK. | established: powered attention null | `postsoftmax.json`, `band_mass.json`, `fig_report_data.json` |
| C14 | A dyad-anchored nucleosomal sequence signal exists at ≈**10.298**, not ≈10.630, and does not explain the multispecies carrier. | established: biological source distinction | `dyad_real.json`, `dyad_wavelength.json` |
| C15 | Local splice-junction phasing exists, but the short support cannot resolve 10.298 from 10.630; the long-intron arm is a powered null and the long-exon arm is underpowered. | bounded | `splice_profile.json`, `splice_phasing.json`, `splice_long.json` |
| C16 | The masked-donor coupling is mediated by a low-rank, specific direction in the **final residual** (the "information object" `U_info`). Patching only the top-8 clean−corrupt directions into the corrupt run recovers **0.92** of the coupling (top-1 **0.74**) through the exact MLM head, versus **≤0.05** for size-matched random rank-k subspaces (add-one **p = 0.005**); necessity mirrors it. Injecting a no-donor-context baseline collapses the **+4.65**-nat donor margin to the base-rate floor (R8 alive-check). Anchors the output end of the L3H1→donor-logit trace. | established: output readout, size-matched null | `donor_anchor.json`, `preregistrations/PREREGISTRATION_DONOR_ANCHOR.md` |
| C17 | **L3H1 delivers `U_info`.** Restoring L3H1's clean output into the corrupt run recovers **0.83** of the coupling (matching `readout_path`), and **0.93** of that routes through `U_info` (patching only its rank-8 component; random rank-8 **≤0.05**, add-one **p = 0.005**). The delivery is localized to the donor **±5** neighborhood (**0.96** of the routing; elsewhere **0.06**); the restoration direction overlaps `U_info` at **0.905**. The acceptor-dependent `U_info` write is assembled at **MLP3 and MLP4** (MLP5 writes `U_info` acceptor-independently). Joins the forward and backward arms of the L3H1→donor-logit trace. A residual **7%** of L3H1's recovered margin routes outside rank-8 `U_info` (open). | established: mediation/path, size-matched null | `donor_forward.json`, `preregistrations/PREREGISTRATION_DONOR_FORWARD.md` |
| C18 | The layer-4/5 attention that carries `U_info` to the donor resolves into a small set of **transporter** heads. Noising each head to its corrupt value: **L5H11 (0.125), L4H2 (0.122), L4H0 (0.081)** destroy the coupling (gene-bootstrap CI-low > 0.05); noising their **query+key** destroys **≈0** — they carry the `U_info` *content* through **OV/value**, not routing. The relays are **general-purpose** heads (mean-ablation → ~0.51 masked-MLM), with a distributed tail of smaller L4/L5 contributors below threshold. Sanity: L3H1 tops the scan at **0.782**; the scan is selective (3/32). Resolves the `readout_path` L4/L5 blocks into heads and closes the head-level meet-in-the-middle. Attention-pattern geometry of the transporters not yet measured. | established: head-level mediation, selective/size-matched | `donor_heads.json`, `preregistrations/PREREGISTRATION_DONOR_HEADS.md` |
| C19 | **The causal signature of `U_info`: paired-intron consistency + donor-identity confidence, and PHASE-INDEPENDENT.** Partner-specific (~37×: **+0.97 [0.83,1.11]** vs distance-matched non-partner AG **+0.03 [0.01,0.05]**), donor-specific (~74×: **+0.96** vs non-site masked G **+0.01 [−0.01,0.04]**), strand-symmetric. **Flat across *both* intron length mod 3 (1.05/0.84/0.99) and annotated splice phase (1.06/0.93/0.99)**, ruling out raw cross-site alignment *and* biological boundary-phase representation. Modestly coding-enriched (coding **+1.01** vs noncoding **+0.62**, overlapping CIs) but present on noncoding boundaries. Reading: a donor GT is scored more probable when its *specific* paired acceptor is intact — site **consistency**, not reading frame. All seven prompt conditions measured (CDS/phase bridge, `gfmcd.splicing.phase`). | established: behavioral signature | `donor_signature.json`, `preregistrations/PREREGISTRATION_DONOR_SIGNATURE.md` |
| C20 | **L3H1's acceptor read is partner-IDENTITY-specific, and its address (key) and content (value) are separable.** L3H1's masked-donor query prefers the intron-mate partner over a distance/phase-matched generic AG at QK (**S_QK +3.60 [3.37,3.80]**), attention (**S_A +0.55 [0.51,0.59]**) and behaviour (**S_Y +0.87 [0.75,0.98]**); the salience-controlled comparison against a non-partner **real** acceptor is **+0.62 [0.46,0.74]** (n=30) and survives its distance confound (real decoys median +19.5nt farther; OLS intercept at matched distance **+0.61 ± 0.08**, slope +0.001/nt, corr +0.08 — identity, not distance-decay). Relocating the partner **key** into the decoy raises L3H1's donor attention to it (no-reloc difference-in-differences **+0.40 [0.37,0.44]**); coupling recovery dissociates address from content (**key-only 0.16 [lo 0.12] > value-only 0.03**). The key+**value** transplant recovers ~equally at decoy/non-site/a⁺ (**0.62/0.62/0.65**, spread 0.024) — architecturally **R15-forced**; **no** content-addressed / position-independent pointer is claimed. L2H15 is a statistically-real but **minor** builder of this preference (mean-ablation removes 11%; retained **0.89**, add-one **p=0.021** vs the 47 upstream heads), i.e. it is built diffusely, distinct from L2H15's larger role vs the *corrupt* partner (C9). **[Interpretation superseded → R20/C22 (2026-07-16).** The empirical `S_QK`/`S_A` preference **survives** (as does its ~15–19% content component); its **partner-identity interpretation is retracted** (R20). A1a shows the preference is dominated by learned **position-embedding geometry** — neutralising the a⁺/decoy position embeddings collapses it to ≈0, incl. the distance-matched arm; the distance-OLS intercept (+0.61) is a sub-component of that geometry. **Behavioural C19 is unaffected** (separate evidence).**]** | empirical preference established at QK/attention (survives); **identity interpretation retracted → R20**; key/value mediation R15-forced; real-acceptor arm modest power (n=30) | `acceptor_address.json`, `preregistrations/PREREGISTRATION_ACCEPTOR_ADDRESS.md`, `archive/DECISIONS_ACCEPTOR_ADDRESS.md` |
| C21 | **L3H1's acceptor-targeting DIFFERENCE is pre-resident in the embeddings, not constructed by layers 0–2; the upstream query pathway instead supplies class-level acceptor attraction *gain* — resolving C20's coupling-vs-preference gap and retiring the diffuse-builder hypothesis.** Stage-wise reconstruction of L3H1's donor-query pre-softmax targeting `S_A` from a mean-resample hybrid residual (post-LN ⇒ intervention, not linear attribution; n=260 generic / n_real=120 real-acceptor, 230 genes; preflight 9/9, three arithmetic memorials EXACT=0.0): the difference is **~fully present at the all-ref floor** (S_J primary **+3.14 vs full +3.04**; secondary **+3.49 vs +3.54**), the prefix-denoising curve is flat with a mid-hump, and reciprocal **leave-one-stage-out isolates the embeddings** as necessary (embed drop **+3.19 / +3.08**; every layer-0/1/2 stage \|drop\|<0.9, several negative). Key provenance rises monotonically (S_K **+2.52→+3.04**); query provenance is high at the generic floor and **falls at layer 2** — because reconstructing the donor query (peak at **mlp2**) raises absolute attraction to **both** a⁺ **and** the decoy vs background (primary a⁺ **+5.59→+7.42**, decoy **+1.41→+2.63**; secondary similar): a **class-level gain** invisible to a⁺−decoy. This resolves C20 (L2H15 supplies **+0.661 nats**, ≈**48%** of the `min_effect ≥ 0.3` conditional coupling baseline, vs only 11% of the a⁺−decoy attention preference — different denominators, different rungs) **[R29: the previously printed "≈66% of coupling" was a unit error — 0.661 is an absolute nats difference, not a fraction]**: layer-2/L2H15 supplies acceptor-attraction **gain**, not identity discrimination. `S_A` is a mechanistic intermediate; the primary arm's distance confound (median **+19.5nt**) and the secondary arm's site-strength conflation mean A0 does **not** adjudicate identity-vs-geometry — that is the token-vs-position next step. | established: QK/attention provenance (sufficiency prefix + reciprocal necessity), confirmatory; **does not establish identity**; ckpt `2ad91428` | `a0_targeting_provenance.json`, `preregistrations/PREREGISTRATION_A0_TARGETING_PROVENANCE.md`, `DECISIONS_A0.md` |
| C22 | **L3H1's acceptor-targeting difference is POSITIONAL GEOMETRY, not local content — and lives entirely within the 4-token key window.** A shared-reference token-vs-position 2×2 on the embedding (neutralising word vs position at the a⁺/decoy key windows so they become identical on the ablated dimension; confirmatory n=260 generic / n_real=120, 225 genes; preflight 8/8, `shared_ref_neutralises`=0.0 EXACT): **position_effect ≫ content_effect**, CI-separated, in **both** arms — primary **+2.72 [2.22,3.15]** vs **+0.40 [0.18,0.62]**; secondary **+3.59 [3.37,3.81]** vs **+0.68 [0.57,0.79]**. Equalising the a⁺/decoy positions collapses the preference to **≈0** (pos_neutral −0.02, CI incl 0) even with content clean → position is necessary **and nearly sufficient**; local within-window content is a **modest but reproducible** contribution (~15–19% of full, +0.40/+0.68), position-synergistic (interaction ≈ content_effect). The **floor is 0.000 exactly** (both-neutral): **no directly recoverable contribution to this L3H1 targeting score lies outside the 4-token key window at this stage** — this does *not* prove PPT/branch-point info never reaches the mechanism (earlier heads could compress nonlocal context into these local residual vectors before L3H1 reads them). Holds in the **distance-matched** secondary arm ⇒ **position-embedding geometry** (distance-decay ruled out; **absolute vs relative vs mod-3 not yet distinguished** — needs a joint donor+acceptor translation test, deferred). **Undercuts C20's content/identity reading** (→ R20); does **not** touch behaviour-level partner-specificity (C19). `S_A` is a mechanistic intermediate; no identity claim. | established: QK/selection provenance (2×2, confirmatory); **position-embedding geometry, non-identity**; ckpt `2ad91428` | `a1_token_position.json`, `preregistrations/PREREGISTRATION_A1_TOKEN_VS_POSITION.md`, `DECISIONS_A0.md` |
| C23 | **L3H1 binds a positional ADDRESS to a partner-specific value PAYLOAD — neither alone recovers the coupling, only jointly.** Reconstruct-inject factorial on L3H1's donor output Σⱼ A_dj V_j (address A = clean/corrupt attention row × payload V = clean/corrupt values), injected donor-only and read as the native margin recovery (confirmatory n=300, 261 genes, coupling 0.96; preflight 7/7, reconstruction & inject-identity memorials **EXACT=0.0**): **address_only +0.01 [−0.02,+0.04]** and **payload_only +0.05 [+0.03,+0.07]** each recover ~nothing, but **joint +0.37 [+0.33,+0.40]** (≈ L3H1's C17 restore for its own channel) — a near-pure **interaction I_{A×V} = +0.30**. Decisive same-row subtest: attention FIXED clean, corrupt→clean values recovers **+0.35**; attention fixed corrupt, **+0.05** — the values carry the discriminating payload but deliver **only** under the correct positional address. A **single** random rank-8 value subspace recovers **+0.18** (~half) — a high generic baseline, so "the payload is distributed" is **provisional** pending a random-subspace *distribution* (A2b), not one draw. **Scope:** joint +0.37 is a substantial **partial** mechanism (L3H1's own channel is a partial mediator, C17) — **not** the complete source of partner-specific behaviour, and **not yet** the payload's *meaning* (A2b localises it to `U_info` vs random). **Reconciles A1a/R20 with C19**: position-embedding geometry routes L3H1 (*where* it reads), the value there carries the partner content (*what* it retrieves), and partner-specific behaviour needs both. Behaviour-level; the address–payload split of L3H1's **coupling** contribution, **not** attention identity (R20 stands). Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level factorial (reconstruct-inject, confirmatory); **address–payload binding (partial mechanism)** | `a2_address_payload.json`, `preregistrations/PREREGISTRATION_A2_ADDRESS_PAYLOAD.md` |
| C24 | **L3H1's partner payload is LOCALIZED to the 4-token acceptor window, and — relative to the preregistered rank-8 `U_info` — is NOT concentrated in that subspace; L3H1 performs a representation *conversion* (broad acceptor-local value → compact donor-side `U_info`).** Residual-patch payload localization (post-LN value linearity `V=W_V·r` ⇒ exact analytic reconstruction with the clean attention row fixed; confirmatory n=300, 256 genes; preflight memorials exact, `value_linearity` 1.9e-6): restoring the clean values at the **acceptor window alone** recovers **+0.38 [0.35,0.42]** (≈ A2 joint +0.37 — the payload is acceptor-local). The `U_info` component recovers **+0.06 [0.03,0.09]**, which is **not** above an **energy-matched random rank-8 distribution** (E[rand] +0.05, range [0.03,0.07] brackets R_U, percentile 0.80, add-one **p=0.22**); the complement `(I−P_U)` recovers ~all of it (**perp +0.37**) and necessity is negligible (**N_U +0.011**, noising loss +0.009). Acceptor-specific (nonsite = floor). *(Descriptive fractions are non-additive and denominator-dependent — raw sufficiency `R_U/R_full ≈ 16%`, raw necessity `N_U/R_full ≈ 2.9%`, floor-subtracted marginal `(R_U−floor)/(R_full−floor) ≈ 4.5%`; the denominator-free facts above carry the claim.)* **Scoped conclusion:** the acceptor-side payload is **distributed *relative to* `U_info`** and is not concentrated in that rank-8 subspace; its **intrinsic dimensionality is unmeasured** (a privileged low-rank subspace could still exist in the 504-d complement — needs a rank sweep / independently-derived `U_source`). **`U_info` is NOT redefined:** it remains the valid DELIVERED donor-side format (C16/C17); A2b shows the same basis is not privileged at the *acceptor source*, revealing a representation change across the head. Behaviour-level; R20 stands. Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level payload localization (confirmatory); **`U_info` not privileged at the acceptor; payload acceptor-local + distributed-relative-to-`U_info`; L3H1 converts representation** | `a2b_payload_localization.json`, `preregistrations/PREREGISTRATION_A2b_PAYLOAD_LOCALIZATION.md` |
| C25 | **L2H15's coupling contribution acts THROUGH L3H1's ADDRESS channel (attention), not its payload — it amplifies/routes acceptor-attraction *gain* but does NOT originate the discriminative positional geometry (that is pre-resident in the embeddings, C21).** Reconstruct-inject L3H1's donor output `A·V` with the address `A` (attention row) and payload `V` (values) drawn from **L2H15-intact vs mean-ablated** runs, injected into the L2H15-ablated forward, reading native **coupling rescue** (confirmatory n=300, 246 genes; reconstruction & baseline memorials **EXACT=0.0**; L2H15 lift `Y_intact−Y_ablated` +0.43): restoring L3H1's intact **attention** rescues **+0.454 [0.39,0.53]** while restoring its intact **values** rescues only **+0.019 [0.01,0.03]**; joint **+0.479**, interaction **+0.006** (no binding) → **~95% of L2H15's donor-channel contribution is the ADDRESS, ~4% the payload.** This **assigns L2H15's +0.661-nat (≈48% of the `min_effect ≥ 0.3` conditional baseline) coupling role [R29]** (l2_accounting/C9): L2H15 shapes L3H1's **positional attention address** (coheres with A1a/R20 — that attention is position-embedding geometry) and does **not** supply the acceptor payload (which is the acceptor's own local, distributed representation, A2b/C24). Donor-only injection (joint +0.48 vs l2_accounting all-position 0.959 ⇒ L3H1's donor output carries ~half of L2H15's contribution; rest via the donor±5 delivery, C17); the address/payload split is a clean property of that channel. **[Wording tightened → C21 reconciliation (2026-07-16): the causal split (~95% address / ~4% payload) STANDS; the original headline "L2H15 builds the address" is corrected to "acts through / amplifies the address channel." C21 shows the discriminative a⁺−decoy geometry is pre-resident in the embeddings, and L2H15 supplies class-level attraction *gain* (11% of the a⁺−decoy preference; **+0.661 nats ≈ 48% of the conditional coupling baseline — R29**) — it routes/amplifies the address, it does not originate the geometry. No R-number: the empirical attribution is unchanged, only the interpretive verb.]** Behaviour-level; R20/C24 stand. Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level attribution (confirmatory); **L2H15 = address-channel gain/router (not the origin of the geometry — C21)** | `a3_l2h15_address_payload.json`, `preregistrations/PREREGISTRATION_A3_L2H15_ADDRESS_PAYLOAD.md` |
| C26 | **The donor-side `U_info` is a STABLE, generalizing model representation — validated, not redefined.** Independent 50/50 **disjoint gene-split** derivation of the C16 object (top-k SVD of the clean−corrupt final donor residual; confirmatory n=400, 320 genes, 199/201 split; `margins_fn` memorial EXACT=0.0): **subspace overlap** `subspace_overlap(U_A,U_B)` ≫ random at every rank (top-1 **0.970** vs 0.002 null; rank-8 **0.630** vs 0.016); **held-out sufficiency** (derive on A, patch on held-out B) **+0.875 [0.853,0.901]** ≈ in-sample +0.911 ≫ random +0.014; **held-out necessity +0.916**. **Reproduces C16** (in-sample rank-8 +0.911 ≈ 0.92, top-1 +0.743 ≈ 0.74). Rank curve: recovery is dominated by the **top-1–2 directions** (+0.73/+0.78) and rises smoothly to +0.94 at rank 32 (no sharp saturation — rank 8 is a reasonable, not magic, choice at +0.91); overlap is highest at low rank (dominant structure rock-solid, rank-8 tail more sample-dependent — expected SVD behaviour, still ~40× the null). **Strengthens C16–C18** (and consistent with C24: `U_info` is the stable *donor-side delivered* representation, not the acceptor source). `U_info`'s definition is unchanged. Behaviour-level. Model checkpoint sha256 `2ad91428…0149d`. | established: donor-side `U_info` stability/generalization (confirmatory); **validation, no redefinition** | `a5_uinfo_replication.json`, `preregistrations/PREREGISTRATION_A5_UINFO_REPLICATION.md` |
| C27 | **The donor circuit is a fully-necessary DELIVERY SPINE, not a faithful sparse generator — complement-noising faithfulness is ≈0 while its clean outputs are sufficient, so the coupling COMPUTATION is DISTRIBUTED or INTERACTION-DEPENDENT at current node resolution.** Node-level activation patching, scrub-**to-corrupt** (post-LN ⇒ patch OUTPUTS, downstream recomputes; two arithmetic memorials `empty→corrupt` / `full→clean` **EXACT=0.0** **[correction 2026-07-16, verify-workflow: these two are EMBED-CLAMP-DOMINATED — `clamp_embed` overwrites the embedding at all positions (corruption is a same-length AG→CC substitution, identical mask), so patching all→source makes the forward bit-identical to the source run regardless of the head/MLP patches. They validate the embed-clamp + correct clean/corrupt caches, NOT head/MLP node-cover completeness. Completeness rests on the ARCHITECTURAL fact that {embed, all heads, all MLPs} = all residual writers, plus the indirect head/MLP-patch validation in the reconcile run; a cover miss would leave a node CLEAN → INFLATE F, so F(C)=0.005 is conservative. cf. R7 (reject architecture-forced answers)]**; confirmatory n=200, 179 genes, coupling 0.94): the 10-node circuit `C` (embeddings + L1H12/L2H15/MLP2 + L3H1 + MLP3/4 + transporters L4H0/L4H2/L5H11) is **NECESSARY** — but `N(C)=1.00` (CI [1.0,1.0]) is a **FORCED IDENTITY** (embed∈C ⇒ clamping embed to corrupt makes the forward == corrupt; not a measured necessity, cf. R7), so the meaningful figure is **`N(C\embed)=0.989`** (corrupting the compute spine alone, corruption **not** applied at the embedding, still destroys 98.9% — a genuine head/MLP-patch result, but **POST-HOC** from the `--reconcile` diagnostic outside the frozen protocol) — but **NON-FAITHFUL** under complement-noising: keeping `C` clean while scrubbing everything else to corrupt recovers only `F(C)=0.005 [0.003,0.008]` (**residual 99.5%**), above the size-matched random-circuit null (p95 0.0001, add-one **p=0.010**) but trivially small; **no** single omitted node recovers it (completeness scan max add-one gain **0.005** over 93 non-circuit nodes) and random same-size circuits are equally non-faithful ⇒ **no sparse node set of this size is faithful**. **Not a contradiction with C16/C17:** injecting the circuit's **clean OUTPUTS** (corruption kept at the embedding) recovers **`D(C\embed)=0.986 [0.982,0.991]`** — the delivered representation is **sufficient-when-injected**, but the circuit cannot **reconstruct** it from a scrubbed network. **Reading:** delivery/readout is **sparse & sufficient-when-injected** (`U_info`, C16–C18); the coupling **generation** (principally building the acceptor-local, *distributed* payload L3H1 reads, C24) is **DISTRIBUTED or INTERACTION-DEPENDENT at this node resolution** — A6 rules out the proposed sparse generator **and** single-node omissions, but **not** a structured jointly-necessary multi-node module; **irreducible diffuseness is NOT established** (→ **A7/C28: RESOLVED — acceptor-local + cross-layer L0–L2 synergistic, a structured jointly-necessary module, NOT broadly diffuse**). **Quantifies the completeness C7 flagged as unmeasured** (general-damage confound **AVOIDED** by scrub-**to-corrupt** — not mean-ablation, sidestepping R12/R13 — though the partial-scrub state's health is **noted-not-gated**, `mlm_health` recorded only at the clean baseline; `D(C\embed)=0.986` is post-hoc and largely expected from C17/C18). Minimality is **uninterpretable at F≈0** (reported, not claimed). Scope: NODE-level, complement-noising, this circuit size; the minimal *faithful* set size — and whether the missing generation is localized-but-multi-node vs broadly diffuse — is unmeasured (A7 localization → scoped edge scan → backward-pruned size curve). Behaviour-level; no prior claim redefined, `U_info` untouched. Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level circuit faithfulness/necessity (confirmatory); **necessary delivery spine, non-faithful sparse generator; generation distributed-or-interaction-dependent (node resolution)** | `a6_circuit_faithfulness.json`, `a6_reconcile.json`, `preregistrations/PREREGISTRATION_A6_CIRCUIT_FAITHFULNESS.md` |
| C28 | **The distributed generation CONVERGES LOCALLY at the acceptor window and is built by a SERIALLY INTERDEPENDENT L0→L1→L2 module — not broadly diffuse, and (for this channel/stage) not imported from the branch-point/PPT or donor regions.** Position×layer localization from the A6 complement-noising baseline (scrub-**to-corrupt**; four plumbing memorials `empty→corrupt`/`full→clean`/`maskFalse→clean`/`state_clamp_identity` =0.0 **[R21: `empty→corrupt`/`full→clean` are embed-clamp-dominated (embed-plumbing only, NOT cover-completeness); `maskFalse→clean`/`state_clamp_identity` ARE valid (mask machinery + layer-input pre-hook); the writers-open head/MLP region patching is validated INDIRECTLY via region-specificity, not an exact memorial]**; confirmatory n=200, 179 genes): the **state-clamp ceiling** (clean pre-L3H1 residual at the acceptor window + clean spine) recovers `F_state=0.372` (ceiling 0.365; ≈ the C23/C24 L3H1-channel cap — internal check). Opening the **non-circuit L0–L2 writers at the acceptor 4-token window alone** recovers **`ΔACC=+0.337 [0.30,0.38]` = 92% of ceiling**; donor±5 (`ΔDON≈0`), branch-point/PPT (`ΔPPT≈0`, CI incl 0) and remainder (`ΔREM=+0.025 [0.012,0.037]`=7%) contribute ~nothing ⇒ the payload L3H1 reads is **constructed LOCALLY at the acceptor**, not imported from the branch-point/polypyrimidine region (coheres with C22 "nothing recoverable outside the 4-token key window" and C19 phase-independence). **This PPT/branch-point null is scoped to THIS partner-coupling channel at THIS (pre-L3H1) stage — it is NOT a biological claim that SpliceBERT ignores the branch-point/polypyrimidine tract in splicing generally.** It is built by L0–L2 as a **serially interdependent pipeline**: single-layer `ΔACC` (L0 **+0.047**, L1 +0.009, L2 +0.001) **sum to 0.057 ≪ 0.337 joint** — but this is **serial interdependence (L0→L1→L2)**, *not* three independent nonlinear factors: a later layer cannot function normally on a **corrupted upstream state**, so the near-silence of single layers is the signature of a **pipeline** ⇒ a **structured, jointly-necessary module** (the review's alternative to irreducible diffuseness, confirmed). **Refines C27**: the generation is "distributed or interaction-dependent" in the specific sense of **positionally localized (acceptor window) + serially interdependent (L0→L1→L2)**. Scope: bounds the **generation** feeding L3H1's channel (the 0.37 ceiling); delivery-side distribution (why the full circuit `F≈0`) is separate. **Next:** decision-table row `state-high + ΔACC-high` → edge scan into the acceptor-window residual across L0–L2. Behaviour-level; no prior claim redefined. Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level source localization (confirmatory); **acceptor-local, serially-interdependent (L0→L1→L2) generation** | `a7_source_localization.json`, `preregistrations/PREREGISTRATION_A7_SOURCE_LOCALIZATION.md` |
| C29 | **The acceptor-payload generator is a serial L0→L1→MLP2 MLP-backbone pipeline (held-out confirmed); L2 attention is not REQUIRED for payload generation under the block intervention, MLP2 has a DUAL address+payload role, and L2H15 is payload-null while remaining essential on the ADDRESS path (C25).** *(All `N`/`P`/`R` below are **conditional causal effects** — leave-one-out necessity / normalized recovery — **NOT additive fractions** of the payload; a serial chain makes multiple stages each individually necessary, so they do not sum.)* Backward-pruned, scaffold-aware exact path search through L3H1's **value path** (address held clean via reconstruct-inject; post-LN linearity **`value_linearity`=2.7e-6** (the REAL memorial — validates the reconstruct-inject on the measurement path; A8's `empty→corrupt`/`full→clean` scrub-cover checks are R21-vestigial and off-path here); C24 anchor reproduced `full_clean=+0.354`; confirmatory n=220, 205 genes, discovery/held-out gene split 112/108): opening the acceptor-window L0–L2 non-circuit writers recovers `M'(A)=+0.346` above corrupt over a scaffold baseline `M'(∅)=+0.057`. **Leave-block-out conditional necessity (replicating on held-out H):** **L0** `N=+0.895 [H 0.87,0.92]` and **40% sufficient alone** (`R_single=0.403`) — the origin; **L1** `N=+0.528 [H 0.43,0.57]` but near-silent alone (0.057) and weak with only L2 downstream (`P=0.095`) — a **necessary refiner requiring L0 upstream**; **L2 attention `N=−0.031` (H-CI incl 0) — not required for payload generation under the block intervention** (L2H15 stays essential on the separate ADDRESS path, C25). The **backbone is the MLPs** (individual leave-one-out `MLP2 +0.893, MLP1 +0.393, MLP0 +0.324`; strongest attention feeder **L0H7 +0.352**), consistent with L0 attention → MLP0 → MLP1 → **MLP2**. **MLP2 is dual-role** — the C21 query/address assembler is **also** the dominant payload relay (`N=0.89`) — while **L2H15 (the C25 address builder) is NOT a payload node** (`N=+0.001 [−0.008,0.011]`): address (L2H15/attention) and payload (MLP2) are **separable within layer 2**. **Method note (validates the group-test design):** 4 individual L2 heads flag load-bearing despite the L2 block being unnecessary — a within-block **interaction artifact**; the block/group necessity is trustworthy, singleton scans mislead here. **Held-out minimal graph** `L0+L1+MLP2` recovers `R_H ≈ 1.0 ≥ 0.80` (dropping L2 attention loses nothing on H). Completes the review's program: non-faithful sparse circuit (A6/C27) → acceptor-local serial generation (A7/C28) → a specific, generalizing serial MLP-backbone generator (A8). Scope: value-path (address held clean), bounds the generation (`0.35` ceiling ≈ C24); the exact head-level graph, head→MLP **edges**, and the payload's **semantic identity** are follow-ups. Behaviour-level; no prior claim redefined. Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level generator path (confirmatory, held-out); **serial L0→L1→MLP2 MLP-backbone; MLP2 dual-role; L2H15 address-only** | `a8_generator_path.json`, `preregistrations/PREREGISTRATION_A8_GENERATOR_PATH.md` |
| C30 | **[R22-corrected — BOUNDED]** **The L3H1 payload is localized to the acceptor window and is NOT explained by the preregistered GC / PPT / k-mer-frequency / positional composition variables (A9b, identifiable). External MaxEnt strength shows only a WEAK NEGATIVE DIRECTIONAL TENDENCY in repeated competitor-matched analyses — sample-variable and not significant under two-way gene-dyadic inference — so the payload's SEMANTIC IDENTITY REMAINS UNRESOLVED (the "encodes splice-site strength" naming is withdrawn, R22).** _The original A9 (a9-val sample) narrative below is DOWNGRADED by R22/A9b and retained only for the record:_ Matched **interchange** interventions (deliver example j's clean acceptor payload through i's **held-clean** address; A8 reconstruct-inject cross-example; `value_linearity`=2.7e-6 exact; frozen `U_info` **projected-onto, not refit**), gated by a **calibration** that PASSED after catching three decision-layer bugs: **planted-positive** sensitivity `r=0.9998` monotonic **and** **permuted-label FPR = 0.047 ≤ 0.05** specificity. Confirmatory n=240, 207 genes, discovery/held-out gene split. On **held-out** genes, energy-stratified, permutation-calibrated (`|r|>r_thresh=0.302`, monotonic, equivalence ∈[0.80,1.20]): the frozen **model-independent local-4-token MaxEnt contribution (V2b)** names it (`r=−0.51`, eq 0.986) **and** the **model's own acceptor logit (V3)** names it (`r=+0.62`), each with the other competitor-matched; the **full-23nt MaxEnt (V2a) does NOT** (`r=−0.04`) — the payload ignores the intronic/PPT flank, **cross-validating C28's ΔPPT≈0** (local window, not the whole 3′SS). **Not** exact sequence identity (V1). **Caveat:** local motif composition (V4) was **NON-IDENTIFIABLE** (no valid matched pairs) ⇒ a composition confound is not fully excluded (follow-up: looser calipers). V2b/V3 are separately identified but biologically correlated (external strength ↔ model confidence). **Completes the mechanism:** position-embedding ADDRESS (C22/C25) × a serial L0→L1→MLP2 PAYLOAD generator (C29) computing **an acceptor-local payload [R22: NOT established as splice strength; semantic identity unresolved]** → L3H1 binds (C23) → donor-side `U_info` (C24/C26) → sparse delivery (C16–C18) → partner-specific prediction (C19). Behaviour+mediation, value-path; `U_info` unchanged. Model checkpoint sha256 `2ad91428…0149d`. | **[R22-BOUNDED]** established: payload is **acceptor-local + NOT bulk composition** (A9b, identifiable — `CV-R²(S~Z)<0`, common support 53–69>40, two confound gates + dyadic bootstrap PASS); **semantic identity UNRESOLVED**; **weak/directional external-strength tendency only** (H: 4/5 seeded draws negative, 4/4 composition-matched subsets negative; not dyadic-significant); splice-strength naming **withdrawn (R22)** | `a9b_payload_semantics_composition.json`, `a9c_matched_composition_reconcile.json`, `preregistrations/PREREGISTRATION_A9b_PAYLOAD_SEMANTICS_COMPOSITION.md`; orig `a9_payload_semantics.json` (R22) |
| C31 | **The generator's head-level wiring (held-out, value-path): {L0H7 dominant + partially-redundant L0H12/L0H10} → MLP0 (which MEDIATES the L0-head contribution) → MLP1 → MLP2 — a strict serial chain.** Head-level edge scan reusing the A8 value-path metric (address held clean; **`value_linearity`=2.9e-6** exact — the on-path memorial; the `empty→corrupt`/`full→clean` scrub-cover checks are R21-vestigial); confirmatory n=220, 196 genes, discovery/held-out split 110/110. **L0-head resolution (held-out `N_h`):** load-bearing `L0* = {H7,H8,H10,H11,H12,H13}`, dominated by **L0H7 +0.348 [0.30,0.40]** (replicates D +0.374), L0H12 +0.160, L0H10 +0.116 (H8/H11/H13 marginal, CI-low>0); **L0H12 & L0H10 are partially REDUNDANT** (interaction **−0.126**), L0H7 independent. **Head→MLP mediation:** L0-heads-only recovers +0.136 but **+MLP0 → +0.356** (`mediated_by_MLP0`=True) — the L0 attention reaches the payload **through MLP0**, not a direct head→value path. **Strict serial MLP chain:** each MLP necessary on held-out (N_MLP0=0.354, N_MLP1=0.434, N_MLP2=0.88, all CI-low>0), **none sufficient alone** (`MLP*_only`≈0.01) → **L0→MLP0→MLP1→MLP2**. **Refines C29** to the head→MLP wiring. All `N`/interactions are **conditional causal effects, NOT additive** (do not sum; serial chain). Scope: value-path (address held clean, so L2H15's address role is removed by construction); L1/L2 individual head resolution deferred. Behaviour-level. Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level generator wiring (confirmatory, held-out); **L0H7/H12/H10 → MLP0 (mediating) → MLP1 → MLP2 serial chain; L0H12/H10 redundant** | `a10_generator_edges.json`, `preregistrations/PREREGISTRATION_A10_GENERATOR_EDGES.md` |
| C32 | **L3H1's donor→acceptor ADDRESS is PREDOMINANTLY RELATIVE (offset-based) with a MODEST absolute-position component — resolving C22's open question.** Consistent **GLOBAL `position_ids` shift** on a truncated W=640 window (A4 redesign; the mandatory **`intact_S_A_survives`** memorial PASSED — `S_A(0)=+2.938`, every `S_A(Δ) ≥ 0.5·S_A(0)` — so the shift **preserves** the preference, unlike A4's isolated-token clamp which collapsed it; `shift_is_noop`=0.0; truncation↔full corr **0.958**); confirmatory n=1037, 641 genes. **`S_A` SURVIVES** the ±192 nt common translation (+2.9→+3.6 throughout) ⇒ the **relative offset dominates**; but it is **NOT invariant**: `E_common(Δ)` is significant at 5/6 shifts, **U-shaped** (min near Δ=0, rising at both extremes: −192 +0.32, +192 +0.70), slope **+0.0012/nt [0.00102,0.00138]** (CI excludes 0), `max|E|/|S_A(0)|=0.237` ⇒ a **modest (~24%) absolute-position modulation**. **Caveat:** the U-shape rising toward both PE-table extremes may be partly **PE-edge artifacts** at `|Δ|=192`; the interior trend (+64/+128) is the cleaner absolute evidence. Raw PE-QK Toeplitz residual **0.56** (descriptive proxy; strong non-offset structure, ignores layers 0–2). **Verdict corrected** from the script's binary "ABSOLUTE" (thresholds trigger but overstate — `S_A` survives). **Resolves C22**: the partner search is primarily distance/offset-based with a modest position modulation. Sibling of the payload arc (A9/C30 = *what*; this = *where*). QK/selection; no behavioural claim; no prior claim redefined. Model checkpoint sha256 `2ad91428…0149d`. | established: QK/selection position geometry (confirmatory); **predominantly-relative address + modest absolute component (A4 redesign, survival-memorial-gated)** | `a11_address_geometry.json`, `preregistrations/PREREGISTRATION_A11_ADDRESS_GEOMETRY.md` |
| C33 | **The coupling is IRREDUCIBLY DISTRIBUTED — a gradual faithfulness–size curve with NO knee — but the identified circuit is the structured CORE at the top of an independent importance ranking.** Backward faithfulness–size curve (rank nodes by F-loss removed **from the full model** on split D — well-identified backward, unlike forward at F≈0; keep-top-k curve on **held-out** H; reuses A6's faithfulness scrub; R21: `F(ALL)=1`/`F(∅)=0` are embed-forced anchors); confirmatory n=200, 184 genes, D/H 102/98. **Node counts (of 103) for held-out faithfulness:** 50%→**34**, 80%→**56**, 90%→**72**, 95%→**80** — a **gradual** curve (F≈1 to `|K|`=80, then smooth decline 72→0.94, 56→0.85, 34→0.61), **no knee** ⇒ **no sparse or mid-size faithful subcircuit**; 90% needs ~70% of the network spanning all layers (core: 65 heads across L0–L5 + all 6 MLPs + embed). **Strong independent cross-validation of A6–A11:** the top-ranked nodes (backward removal-loss, *no knowledge of the hand-built circuit*) ARE the identified mechanism — **embed #1, MLP2/3/1/0/4/5 #2–8, L3H1 #4, generator L0H7/H12/H10 #9/11/13, transporters L4H2/L5H11/L4H0 #10/12/15**. **Reading:** a structured CORE (the identified circuit) embedded in a broadly distributed ~72-node faithful core — reconciling A6 (non-faithful sparse circuit) with the A7–A11 structure. **Quantifies A6's "distributed"** as irreducibly distributed *with a hierarchy*. Caveat: static backward-rank = **upper bound** on the minimal faithful set (true greedy could shave a few via interactions; the gradual shape is robust). Behaviour-level, node granularity. Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level faithfulness–size curve (confirmatory, held-out); **irreducibly distributed (no knee); identified circuit = top-ranked structured core** | `a12_faithfulness_size.json`, `preregistrations/PREREGISTRATION_A12_FAITHFULNESS_SIZE.md` |
| **IC1** | **[DIAGNOSTIC — outcome-aware VERIFICATION, NOT confirmatory]** **Distal intron interstitial ARRANGEMENT (dinuc-composition-controlled ordering) contributes LITTLE MEANINGFUL, PARTNER-SPECIFIC donor–acceptor coupling** in the conservative-constructible minimal-intron subdomain (ACC_UP=70, L≳107). Behaviour-first 2×2 DiD `ΔC` on a **conservative-scramble** primary arm (cryptic-site change bounded BY CONSTRUCTION; anchored/unconstrained secondary), signed cluster-mean, BCa CI, **deterministic reproducible cohort** (259 introns / 256 clusters; sorted-set shuffler + pinned hashseed), MDE 0.029: **`ΔC̄=+0.032 nats`, CI [+0.012,+0.053]** — statistically detectable (excludes 0) but **SUB-SESOI (<0.05)** ⇒ *small-but-detectable*, NOT material. **NO meaningful partner-specific effect:** the paired contrast `ΔC_specific = ΔC_partner − ΔC_sham = +0.006`, CI95(BCa) **[−0.011, +0.024] INCLUDES 0** — the partner-specific component is indistinguishable from zero (this paired contrast, not "sham≈mean", is the rigorous support). Cryptic-competition arm-contrasts NULL. All gates pass (corrected incremental-within-state OOD noninferiority; on-path positive control). **Status caveat:** ΔC was seen on the v7 run before the v8 reproducibility + verdict fixes, so this is a deterministic VERIFICATION, not a preregistered confirmatory null (which would need a genuinely untouched disjoint cohort). Coheres with the local-window generation picture (C28/C30). **Next:** IC2 (branchpoint/PPT + acceptor-proximal), IC5 (competing acceptors). Amendments A1 (outcome-blind OOD-gate correction), A2 (reproducibility + verdict fixes). | diagnostic/verification (outcome-aware): **distal arrangement is NOT a meaningful partner-specific coupling driver; small sub-SESOI nonspecific effect only** | `ic1_distal_intron_necessity_v8.json`, `ic1_pilot{,2,3,4_ood}.json`, `test_ic1_determinism.py`, `preregistrations/PREREGISTRATION_IC1_DISTAL_INTRON_NECESSITY.md` (A1,A2); v6/v7 preserved |

| C34 | **The period-3 positional carrier's causal contribution to donor coupling is SPATIALLY DISTRIBUTED across the donor–acceptor span, NOT localized to either site — restoring the carrier only at the local site windows recovers ~2–4% of coupling; broad-span restoration is required.** Position-selective restoration (P3.2): in the period-3-deleted model, add back the carrier delta (`P_intact−P_deleted`) at per-example windows and measure normalized coupling recovery `R_C=(C_a−C_del)/(C_intact−C_del)` (ratio-of-means, gene-clustered CI; **discovery/confirm gene split** n=160, 144 genes, 77/83). Plumbing memorials PASS (element-wise `P_hybrid` construction 1.5e-8; restore-none==deleted 0.0; restore-all→`R_C`=1.00 forced; MONOTONE dose-response). Deletion collapses coupling (`C_intact` 1.39/1.51 → `C_del` 0.05/0.08 = 96.7%/94.5%). **M1 (`C_del/C_intact ≤ 0.05`) is a preregistered Global-PASS precondition (prereg §8): PASS on discovery (3.3%) but FAILED on confirm (5.5%>5%)** ⇒ the confirm split is **gate-qualified / verification-level, NOT a clean preregistered confirmation (R24)**; the distributed-locus pattern is identical on both splits and the denominator `C_I−C_D` stays large (robust), but robustness does not retroactively pass a failed preregistered gate. **Local-window restoration recovers little:** donor ±6 `R_C`=+0.016/+0.019 (CI-exclude 0), acceptor ±6 +0.016/+0.007, both ±6 +0.035/+0.041 — all **≪ the SESOI recovery threshold** (`R_C`≥0.2). **Specificity nulls ≈0** (mod-3-phase-matched irrelevant window +0.002/+0.003; decoy-acceptor window +0.002/−0.002) ⇒ the small local recovery is **site-specific, not a hybrid-`P` edge artifact**. **Recovery requires a broad span:** the both-window dose-response rises monotonically to `R_C`≈0.41/0.50 only at **±32 nt** (covering most of the 60–150 nt inter-site region). Health-gated (kept ≥0.82 all arms). **Reading:** the carrier is read as a *distributed positional coordinate across the span*, not at two point-addresses — the P3.2 **"distributed upstream organization"** outcome (Endpoint B for the locus), a **well-powered positive characterization** of distribution (tight CIs, clean nulls), not an underpowered null. **Coheres with C32** (address predominantly RELATIVE offset — a relative code needs the carrier along the span) and **C33** (coupling irreducibly distributed, no knee). Contrast: the PAYLOAD is acceptor-local (C28/C30) while the positional CARRIER contribution is span-distributed. QK/positional through behaviour (coupling); **no identity or binding claim** (R23). Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level carrier locus; **all gates PASS on discovery**; the pattern **replicates on confirm but that split is GATE-QUALIFIED — preregistered M1 gate failed (5.5%>5%, R24) → verification-level replication, not a clean preregistered confirmation**; **spatially distributed carrier (not site-local); Endpoint-B locus** | `results/p3_carrier_locus.json`, `preregistrations/PREREGISTRATION_P3_CARRIER_CAUSAL_ROLE.md`, `redteam/REDTEAM_SYNTHESIS_P3_4lens.md` |
| C35 | **The period-3 carrier's contribution to donor coupling routes through L3H1's ATTENTION PATTERN (the positional address — *where it looks*), NOT through carrier-dependent changes in its VALUES (L3H1's own value vectors — *what that head collects at the acceptor*): a clean, well-powered A-vs-V dissociation.** L3H1 attention×value factorial (P3.3, reframed per R23 — axis "A vs V", **not** address/payload, since V carries position, R15): on the period-3-DELETED base, patch head-1's query,key to intact-carrier captures (⇒ attention pattern `A_I`) and/or value to intact (⇒ `V_I`) via `capture_proj`/`patch_proj` (module runs natively — one `W_O`, exact); read native coupling. **Plumbing anchors EXACT 0.0** (M4 diagonal; off-diagonal *native* value-overwrite; head-identity — pins head-1=L3H1). Per-example **paired** contrasts, gene-clustered CIs, disc/confirm gene split (n=120, 114 genes, 62/58). **Restoring L3H1's intact-carrier ATTENTION recovers coupling** (routing main **+0.257 [0.172,0.351] / +0.284 [0.192,0.386]**, disc/conf, CI≫0) **while restoring its VALUES recovers ≈0** (value main **−0.011 [−0.020,−0.000] / −0.005 [−0.013,+0.003]**). Cells (A,V): DD +0.03/+0.03, ID +0.30/+0.31, DI +0.03/+0.02, II +0.42/+0.46. **The value-main is a GENUINE (non-forced) sub-SESOI null** — |−0.011| ≪ SESOI 0.05, ~24× smaller than routing and **wrong-signed** (discovery marginally excludes 0 at −0.011 [−0.020,−0.000]; untouched confirm CI includes 0 [−0.013,+0.003]) ⇒ practically zero / equivalence, not a positive value effect. Restoring `V_I` *could* have moved coupling if L3H1's value vectors were carrier-dependent; it does not ⇒ **the carrier's coupling contribution is NOT mediated by carrier-dependent changes in L3H1's values** — a within-intervention statement that does **not** exclude carrier effects on payload *construction* upstream (the carrier feeds the L0→MLP2 generator, C29/C30; R24). **`Δ_bind`=+0.136/+0.157 is CI-positive but ARCHITECTURE-FORCED (R23)** — the bilinear `O=A·V` makes the 2×2 interaction the identity `(A_I−A_D)(V_I−V_D)`, non-zero for any A-and-V-moving perturbation; **reported, NOT interpreted as binding** (a striking confirmation of the pre-run red-team's foresight — the original design would have mis-reported this as binding). **Reading:** the carrier is a positional ROUTING coordinate read as L3H1's attention address — it shapes *where* L3H1 looks, not *what* it retrieves. Coheres with C22/C25/C32 (L3H1 attention = position-embedding geometry) and C24/C28/C30 (payload acceptor-local, generated by the L0→MLP2 backbone). Global health unchanged across cells (single-head edit); local validity from the exact M4 anchors (global health necessary-not-sufficient here). Behaviour/mediation (coupling); conditional mediation only. Model checkpoint sha256 `2ad91428…0149d`. | established: behaviour-level A-vs-V routing decomposition (confirmatory, disc/confirm split); **carrier routes through L3H1's ATTENTION (routing), NOT carrier-dependent changes in L3H1's values (a genuine null, within-intervention — not a global payload-construction claim, R24); `Δ_bind` forced (R23)** | `results/p3_av_decomposition.json`, `preregistrations/PREREGISTRATION_P3_CARRIER_CAUSAL_ROLE.md` |
| C36 | **Nucleotide-invariant local carrier surgery cleanly confirms phase-aligned use of SpliceBERT's learned P3 hidden direction for donor–acceptor coupling in CDS blocks.** All 11 frozen instrument gates pass. Across 943 P3-analysis-untouched genes, the within-gene CDS-minus-upstream-intron modulo-recovery estimand is **Ψ_C=+0.141 [0.110,0.172] nats**, above the 0.05-nat SESOI (MDE80 0.045). In-register shifts `{3,6,9}` preserve CDS coupling at ≈0.757 while adjacent off-register shifts reduce it to ≈0.614; the identical intronic surgery changes coupling by only ≈0.002. The learned P3 direction exceeds every one of 19 pre-seeded signed-hidden-axis permutations that preserve spatial waveform, Fourier support, full spatial Gram matrix, row norms, block geometry, and perturbation RMS: target +0.135, controls −0.0054..+0.0012, exact add-one rank p=0.05, paired target-minus-control **+0.136 [0.051,0.222]**. The matched 52-pair CDS-minus-non-CDS-exon contrast is **+0.006 [−0.080,+0.093]**, inconclusive; coding enrichment is therefore **not** licensed. The intervention changes no nucleotide, codon, motif, MaxEnt score, coordinate, or length. | established: clean preregistered behavior + learned-direction specificity; coding-context boundary inconclusive | `results/phase-lattice/p3_phase_surgery_confirmation_v1/confirmation_analysis.json`; `PREREGISTRATION_P3_CLOSURE.md` |

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
| R20 | C20's interpretation of the distance-adjusted L3H1 **attention** preference (`S_QK`/`S_A`) as evidence of partner **identity**. | A1a (C22) shows that preference is dominated by learned **position-embedding geometry**: neutralising the a⁺/decoy position embeddings removes it (→ ≈0, incl. the distance-matched arm), while neutralising word content preserves most of it. The distance-OLS intercept (+0.61) is a sub-component of that positional geometry, not proof of non-positional identity. | Report the empirical preference (it survives, as does its ~15–19% content component) but **not** as identity. Distinguish empirical statistic (survives) / identity interpretation (retracted) / behavioural C19 (survives, separate evidence). |
| R21 | The `empty→corrupt` / `full→clean` "exact arithmetic memorials" (A6/A7/A8/A10) were framed as **proving node-cover completeness + head/MLP patch plumbing**, and `N(C)=1.00` (A6/C27) as a **measured necessity**. | `clamp_embed` overwrites the embedding at ALL positions, so with a same-length AG→CC corruption (identical mask) the memorial forward is **bit-identical to the source run regardless of the head/MLP patches** (embed-clamp domination); `N(C)=1.00` is forced by `embed∈C` — both are new instances of **R7** (architecture-forced answers). Found by the 2026-07-16 verify-workflow. | These memorials validate the **embed-clamp + correct caches only**. Node-cover completeness rests on the **architectural** fact that `{embed, all heads, all MLPs}` = all residual writers, plus the genuine **`value_linearity`** memorial (A8/A10, on the measurement path) and the indirect `D(C\embed)`. Report **`N(C\embed)`** (post-hoc) as the real necessity, not `N(C)=1.00`. Empirical findings (F≈0 conservative) are unaffected. |
| R22 | C30's claim that the payload **ENCODES local (4-token) splice-site strength** ("externally grounded, held-out `r=−0.51`"); and the A9 practice of treating **one tag-seeded gene-split** as a sampling-stability test with **permutation-on-correlation** inference. | A9's reported held-out association is **exactly reproducible on the original `a9-val` sample** (`tag="a9-val"` → `r=−0.512`(H)/`−0.323`(D)) — **not an implementation error** — but it is **not stable across independently seeded example draws** and does **not survive two-way gene-dyadic inference**. Across five independently seeded draws the held-out correlation ranged **−0.512 → +0.017**, while the discovery-side effect averaged ≈0; **every composition-adjusted dyadic CI included zero, including that of the original sample**. A9's inference did not adequately account for **sender/receiver gene reuse** across interchange pairs; the D/H split was legitimate held-out generalization *conditional on that sample* but **not a test of sampling stability**. V3 does not rescue the interpretation (circular + split-inconsistent). Found by the 2026-07-17 A9b sweep. | **Do not claim the payload encodes local splice-site strength.** What remains established (A9b): the payload is **acceptor-local and is NOT explained by the preregistered bulk/frequency composition covariates** (GC, PPT, k-mer-frequency, positional) under adequate common support. Repeated competitor-matched analyses show a **weak negative tendency** consistent with local strength, but this is **bounded, sample-variable evidence rather than a semantic identification**. For effect-existence claims require **multiple independently seeded (ideally demonstrably disjoint) draws + two-way dyadic inference**, not within-sample held-out alone. |
| R23 | The **P3.3 "address × payload BINDING" endpoint** — a CI-positive L3H1 attention×value interaction `Δ_bind` as evidence of binding — and the **"independently-constructed payload"** clause (via a `payload_source` estimand). | **`Δ_bind` is architecture-forced:** L3H1's output is bilinear `O=A·V`, so the 2×2 DiD equals the identity `(A_I−A_D)(V_I−V_D)` — non-zero for **any** perturbation moving both A and V, with **no carrier-specificity** (there is an energy-matched null for the main deletion but **none for the interaction**; the same positive `Δ_bind` would arise from a junk band or any random PE perturbation). **`payload_source` is non-identifiable:** no delivery-independent, carrier-uncontaminated observable exists (the carrier feeds the generator; MLP2 is dual address+payload, C29). Caught by the **2026-07-17 4-lens design red-team** (`redteam/REDTEAM_SYNTHESIS_P3_4lens.md`) **before any confirmatory run** — new instances of R7/R15/R21 (architecture-forced answers; position already inside V). | **Do not claim address×payload binding from an L3H1 A×V interaction, nor an "independently-constructed payload," within a carrier-deletion+patching design.** P3.3 is reframed to an *A-vs-V routing decomposition on coupling* (conditional mediation; axis "A vs V", not "address vs payload" — R15), requiring an **importance-matched interaction null** (`Δ_bind` above the E-band distribution) and reporting the bilinear floor. Construction-vs-binding is **non-identifiable here → Endpoint B**. **C23** (behaviour-level reconstruct-inject address×payload binding — a DIFFERENT, non-forced design) is **unaffected**. |
| R24 | Two P3.3/P3.2 framing overreaches: **(a)** C35's clause **"the carrier does NOT build the payload"** — a categorical, cross-intervention reading of the L3H1 value-null; **(b)** describing C34's **confirm split as a confirmation** despite the preregistered gate **M1 (`C_del/C_intact ≤ 0.05`, part of Global PASS, prereg §8) FAILING on confirm (5.5% > 5%)**. | **(a)** The value main effect is a null on **L3H1's own value vectors** under carrier patching; it cannot exclude carrier effects on payload *construction* upstream (the carrier feeds the L0→MLP2 generator, C29/C30). Over-generalizing "L3H1's values are carrier-independent" to "the carrier plays no role in the payload" exceeds the intervention (same empirical-vs-interpretation discipline as R20). **(b)** M1 was frozen into Global PASS; 5.5% residual means deletion left >5% of coupling on confirm, so not every preregistered precondition held on that split — robustness of the denominator does not retroactively pass a failed gate. | **(a)** State C35 **within the intervention**: *the carrier's contribution to coupling is mediated predominantly through L3H1's attention pattern, not through carrier-dependent changes in L3H1's values; this does not exclude upstream carrier effects on payload construction.* **(b)** Report C34 as **all-gates-pass on discovery, gate-qualified / verification-level on confirm**; a clean confirmatory locus needs a disjoint untouched split on which M1 also passes (tighten the deletion, or the `min_effect` conditioning that inflates residual coupling). The empirical findings — distributed locus (C34) and routing≫value (C35) — are **unaffected**. |

> **R2–R3 supporting measurement**, folded in from `src/gfmcd/splicing/rivals.py` before that retracted
> v1/v2 scaffolding was deleted (2026-07-31). The preregistered shortcut table is the measurement that
> killed v1: a **geometry-oracle** adversary reading **zero nucleotides** scored **PMR 1.000** on v1
> (oracle site detector also 1.000; leak-only 0.483; length-only 0.495), which is identification failure,
> not model skill. Under v2 (chr22) the same adversary collapses to **0.076** against a chance rate of
> 0.458, the oracle site detector sits at exactly chance (0.458) and the candidate-index prior at 0.469 —
> the empirical evidence that the v1 forced-choice task was unidentified and that the v2 redesign removed
> the shortcut.

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

## R25 — carrier origin moved from "deferred (needs a new model)" to corpus-attributed (2026-07-23)

**What changed.** MECHANISM items 3–4 previously held that carrier origin was unanswerable on frozen
checkpoints and required training-time evidence (checkpoint trajectories, controlled retraining) or a
new model — i.e. a "reason to move on," deferred to a future project. Two published-checkpoint natural
experiments now license a *comparative* origin attribution without any retraining, so the origin question
is partly closed rather than wholly deferred. New live rows: **C39** (sibling corpus contrast + sibling
band-deletion) and **C40** (cross-model absence + tokenizer-aliasing contrast). C12 amended: the ≈10.63
carrier's corpus origin is now licensed; its general OV/MLM-output function stays open.

**Why the prior wording was too conservative, and its exact remaining limit.** The claim "a frozen
checkpoint cannot identify why training created it" is true for training *dynamics* but too strong for
*attribution*: the SpliceBERT authors published two stage-1 siblings differing only in corpus
(`SpliceBERT.510nt` 72-species vs `SpliceBERT-human.510nt` human-only), which is a controlled natural
experiment already on disk. The remaining limit is honest and stated in C39: this is comparative
attribution (feature tracks corpus), not a trajectory or a controlled-retraining causal proof.

**The load-bearing, non-obvious result.** period-3 is **absent in the human-only sibling** (6.2× vs 435×
median, undetected under a powered floor). Codon structure is present in human coding sequence, so the
naive prior was that period-3 would survive human-only training. It does not: the period-3 clock is a
product of multi-species pre-training, not a generic codon read-out. Corroborated by C40 — period-3 is
absent from single-nt transformer MLMs trained on ORF-free ncRNA (RNA-FM, RNA-BERT) under a powered null.

**A false positive that was caught and must not re-enter.** The cross-model reader scan initially flagged
16 RNA-FM heads "reading" ≈10.63. A frequency control showed this is an envelope/permutation artifact
(n_sig 14 < control-grid max 28; period-3 n_sig = 0). Do not cite RNA-FM as carrying a ≈10.63 reader.

**Terminology note (execution).** The sibling behavioral step is recorded as a Fourier
**band-deletion / carrier-transfer** intervention (identical mechanics to `run_carrier_surgery.py`);
the word "surgery" is avoided in the cross-corpus/cross-model code and prose for these origin results.

**Evidence-ladder placement.** C39/C40 sit at a new column not in the P→Biology ladder above: *corpus
attribution* (which training corpus builds the feature), below Biology (an independently measured
biological source/function). Corpus origin is licensed; a biological function for either carrier is not.

## R26 — SEALED §2's illustrative val figures superseded by the frozen gate + artifact (2026-07-24)

**What changed.** `preregistrations/PREREGISTRATION_SEALED.md` §2 (The behaviour) states the val population
gives **0.973 [0.919, 1.023]** unconditional and **1.386** conditional (a 1.44× inflation). The **frozen §4
Gate A1** and the controlling artifact `results/splicebert_coupling/sealed_val.json` record **0.958
[0.855, 1.059]** unconditional and **1.285** conditional (1.34×). The §2 pair matches neither and does not
reproduce from any artifact; it reads as a pre-freeze / illustrative computation left in the motivation
section before the sealed cohort was frozen (§2 itself already reconciles an even earlier conditional 1.359,
R10). **Rule:** cite the artifact values (0.958 unconditional / 1.285 conditional); §2's 0.973/1.386 are
superseded and must not be quoted as the result. The frozen prereg is retained byte-identical (immutability).

## R27 — A9b primary DML nuisance switched from the preregistered GBR to linear Ridge (2026-07-24)

**What changed.** `PREREGISTRATION_A9b_PAYLOAD_SEMANTICS_COMPOSITION.md` froze the **primary** DML nuisance
as a shallow **gradient-boosted regressor** ("DML nuisance = sklearn GBR (frozen)"), with linear-Ridge named
only as the **secondary/robustness** learner. The shipped `run_a9b_payload_semantics_composition.py` makes
**Ridge the primary** learner (`cv_r2(..., learner="ridge")`; cross-fit residualization uses `"ridge"`)
because GBR produced **negative out-of-fold R²** on the low-dimensional composition Z (overfit). The choice
is sound — Ridge is the honest nuisance and was the prereg's named fallback — but it was **data-driven (made
after inspecting OOF R²) and was not filed as a prereg amendment**. **Rule:** the reported A9b result uses
Ridge; this switch is disclosed here as a post-hoc deviation, with the GBR arm remaining the robustness
check the prereg specified. The A9b empirical conclusion — the payload is acceptor-local and NOT explained
by bulk composition — is unaffected.

## R28 — G1 (C41) planted-positive gate applied as a point check; planted-confound arm not run (2026-07-24)

**What changed.** `PREREGISTRATION_G1_CROSS_EXAMPLE_TRANSDUCTION.md` specified the planted-positive
sensitivity gate as **PASS iff τ is monotone in β AND the fitted slope CI covers 1**, and required a
**planted-confound** arm inside the frozen **Global PASS = AND(...)**. The shipped
`run_g1_cross_example_transduction.py:463` implements planted-positive as **`monotone AND slope > 0`** — a
**point check**, not the interval **slope-CI-covers-1** — and the runner contains **no planted-confound
computation** (only planted-positive). **Rule:** the C41 conclusion (no example-specific transduction of the
C16/C26 final-residual state; a generic readout axis) is reported at **verification tier** with both gate
deviations disclosed; a confirmatory G1 requires the interval-form planted-positive gate and the
planted-confound arm. The frozen prereg and the G1 artifacts are unchanged.

## R29 — "L2H15 ≈66% of coupling" is a UNIT ERROR; the value is +0.661 nats ≈ 48% of the conditional baseline (2026-07-25)

**What changed.** C21 (`:44`) stated "L2H15 ≈66% of coupling, 11% of preference"; C25 (`:48`) stated
"L2H15's ~0.66-of-coupling role" and "(11% of the a⁺−decoy preference, ~66% of coupling)". Two live script
docstrings carried the same reading (`run_a3_l2h15_address_payload.py:6`, "~0.66 of the donor coupling") and
a separate variant (`run_l2_accounting.py:4`, "~0.72 of coupling" — 0.72 is the coupling **retained** after
ablation, not the amount supplied). **The figure is not a fraction.**
`results/splicebert_coupling/l2_accounting.json` (val, n=150) stores
`coupling_supplied_by_L2H15 = 0.6611018248399099`, an **absolute donor-margin difference in nats**:
`run_l2_accounting.py:125` computes `dnm += (cbase − cabl)`, and 1.3840273753801982 − 0.7229255515336991 =
0.6611018238464991. "0.661" was transcribed as "0.66" and then read as "66%". The **fraction** is
0.6611018248399099 / 1.3840273753801982 = **0.4777 → ≈47.8%** (52.2% retained). Two same-construction routes
corroborate: `sealed_test.json` `necessity.heads.L2H15.frac_lost = 0.4765` (test, conditional n=150) and
`faithfulness.json` `necessity.L2H15.frac_lost = 0.5061` (val, conditional n=200). The rescue rows in the same
JSON *are* fractions (denominator `dnm`), which is why "L3H1 head output @all-pos = 0.959" is legitimately 96%
of L2H15's *supplied* coupling — the adjacent 0.6611 is not on that scale.
**Rule:** state it as **+0.661 nats ≈ 48% of the `min_effect ≥ 0.3` CONDITIONAL baseline** — never as a bare
"66% of coupling", and never as a share of the unconditional behavioural coupling (R10). The qualitative
finding is unchanged: L2H15 supplies acceptor-attraction **gain** through L3H1's address channel, not identity
discrimination (C21/C25). **`CLAIMS.md` needs no change** — the compact C21/C25 rows never carried the number.
**Deliberately left carrying the old reading, as historical record:** the frozen A3 preregistration, every
result JSON and manifest, and the running decision log `DECISIONS_A0.md` (`:17`, `:143`, `:255`, `:268`,
`:275`). Those are immutable provenance of what was believed at the time and are to be read through this
entry; do not rewrite them.

## R30 — the ≈10.298 nucleosomal comparator is measured in MOUSE, and the human-spectrum null was over-stated as "absent" (2026-07-25)

**What changed.** Two defects in the same T1 sentence, at `README.md:44` and `docs/ROADMAP.md:69-70`.
**(a) Species.** The ≈10.298-nt dyad-anchored periodicity — the comparator that carries the project's
powered refutation of the chromatin reading of the ≈10.63 carrier — is measured in **mouse ES-cell
chromatin**: Voong et al. 2016 chemical-cleavage dyad map, build **mm9**
(`data/NUCLEOSOME_MAP.md:12,30`; `run_dyad_real.py:2`; `dyad_real.json` `"build":"mm9"`, splits
`chr1,chr2`/`chr3,chr4`, n=10,668,011 dyads). Both documents attributed it to **the human genome**.
`git grep -lIiE 'mouse|mm9|mESC|Voong' -- '*.md'` returned **exactly one file**, `data/NUCLEOSOME_MAP.md`
— the organism appeared in no live prose document, no ledger, and neither prior review.
**(b) Over-claimed null.** Both documents said ≈10.63 is **absent** from the human genome's spectrum.
`corpus_spectrum.json` is an **unanchored, underpowered** assay: `results/phase-lattice/RESULTS_INDEX.md:85`
already says "report its sensitivity limit only", and its own calibration puts the detection floor at
**5–10% modulation** (injected depth 0.05 → Z=1.03, n.s.; 0.10 → Z=1.20, p=0.016). C40's ledger wording is
correct — "no **detectable** ≈10.63 periodicity" — but the two narrative sites hardened it to "absent",
re-committing the error R16 exists to prevent.
**Rule:** always name the organism when quoting ≈10.298, and say **"not detectable above the 5–10%
modulation floor"**, never "absent". **The refutation stands and is unweakened**: the arithmetic reproduces
(10.62993875787851 − 10.297612131759664 = 0.3323), and it does *not* depend on the mouse point estimate
transferring to human at better than 0.33 nt, because 10.630 lies outside the 10.2–10.5 bp DNA-helical range
recorded from the literature. Do **not** write that the refutation is "conserved" across species or
"transfers" — no artifact in this record supports a cross-species conservation claim at the Biology rung.
**No human dyad-anchored measurement exists in this record.** `docs/NOMENCLATURE.md`'s 10.298 row now carries
the organism and is the live anchor for it.
**Sites corrected:** `README.md:44`, `docs/ROADMAP.md:69-70`, `docs/NOMENCLATURE.md:27`, and
`docs/CARRIER_ORIGIN_SYNTHESIS_2026-07-23.md:55-58` — the last found only by the post-edit propagation grep,
not by the review sweep, which is the failure mode this repo has to design against.
**Deliberately left unchanged:** `figures/splicebert/fig_two_carriers.CAPTION.md:64-65` (stamped **T3 —
historical, superseded**; it is a record of what was written then) and `data/NUCLEOSOME_MAP.md`, which was
always correct and is the primary provenance document for the Voong map.

**Addendum (2026-07-25) — the residual gap is now registered, not just disclosed.** Two follow-ups the
first pass left open are closed as far as they can be without new data:
(a) **Standing phrase SP2** (`docs/NOMENCLATURE.md`) now requires the **organism** in the same sentence as
the 0.33-nt qualifier, so the species travels with the number by rule rather than by memory.
(b) **`docs/ROADMAP.md` open problem 6** records the actual scientific gap: *no human dyad-anchored
nucleosome measurement exists in this record*, every dyad script is mm9, and closing it needs an external
human dyad call set on hg38 — not new model compute. The refutation itself is unaffected and is **not**
contingent on cross-species transfer at 0.33 nt, because 10.630 falls outside the 10.2–10.5 bp helical
range recorded from the literature; but the cross-species step is an assumption rather than a measurement,
and it is the most likely line of attack on the project's flagship biological claim.

## R31 — the ≈10.63 carrier's OV/Behaviour evidence was mis-stated in two opposite directions (2026-07-25)

**What changed.** Two errors about the same evidence, pointing opposite ways.

**(a) "Not run" — false.** `docs/EVIDENCE_LADDER.md:36` marked the carrier's OV/mediation and Behaviour
cells **`?`**, and that document's own legend (`:61`) defines `?` as *"the defining intervention has **not**
been run, so there is no evidence in either direction."* `figures/splicebert/EVIDENCE.md:45` stated it in
prose: *"No OV-mediation experiment has been run, and no behavioural effect has been demonstrated."* **Both
interventions ran on 2026-07-23** under B1 (C37, folded into C12), producing
`results/phase-lattice/{carrier_ov_av,carrier_p10_behaviour_confirm,carrier_preflight_r}.json` with 8 rows in
`results/MANIFEST.csv`. A measured null or bound is a **result**, not an absence — the project's own rule —
and the table had no marker for it.

**(b) "Does not write" — over-claimed.** `CLAIMS.md:35` (C12) said the carrier *"routes but does not write,
at the donor readout or generally."* The **donor-readout write test was never computed.** B1 v2 §1
preregistered the OV endpoint as `S_ov` — the `U_info`-projected `suff_recovery` — plus the `Ω` subspace
overlap, each against a covariance-matched null, with a `PP-ovplant` planted-positive sensitivity gate.
`run_carrier_ov_av.py` contains **zero** hits for `U_info`, `S_ov`, or `suff_recovery`. What it measured
instead is a simple **coupling-margin value main** `Mv = DI − DD` over the nine readers: range −0.009 →
+0.013 nats, no head's CI-low exceeding the pre-declared `SESOI_v` = 0.02 (though L2H6's CI upper, 0.0202,
does, and 3 of 9 CIs exclude 0). That is a **sub-SESOI non-detection on the coupling readout at deep
(L2–L4) readers**, with its sensitivity calibration deferred — not a powered equivalence, and not a test of
whether the carrier writes onto `U_info`.

**Rule:** state the two arms separately and at their achieved strength. **OV:** the carrier's readers are
**sub-SESOI on the donor coupling margin**; whether it writes onto the donor readout `U_info` is **OPEN**,
and the B1 prereg's **A-mixed branch — a carrier-specific component written onto `U_info` that is
behaviourally redundant for coupling — remains UNEXCLUDED**. **Behaviour:** deleting the carrier moves
coupling by **−0.033 nats [−0.061, −0.005]** — small, *suppressive*, sub-SESOI, equivalence underpowered.
The general-MLM conjunct of C12 (readers' value-main below the period-3 router-null scale, G ≈ 0.002–0.003
vs router 0.0043) is separately supported and is **not** withdrawn. C12 remains closed-bounded and terminal
on the frozen checkpoint; what changes is the **scope** of "does not write", which was never established at
the donor readout. The evidence ladder gains a **`bounded`** marker so a measured boundary is no longer
recorded as an absence.
**Sites corrected:** `CLAIMS.md:35` (C12), `docs/EVIDENCE_LADDER.md:36` and the marker legend at `:61-67`,
`figures/splicebert/EVIDENCE.md:15,42-46`, `docs/BIOLOGICAL_BRIDGES.md:212-214`, `FINDINGS.md:109-110`,
`figures/splicebert/README.md:83-84,86-87`.
**Deliberately left carrying the superseded reading, as dated record:** `DECISIONS.md:126,192` (a T2
chronology of pivots, explicitly "not a second claim ledger") and `DECISIONS_A0.md`. Read them through this
entry.

**Addendum (2026-07-25) — C12's terminality was itself wrong, and is corrected here.** The first pass at
this entry scoped C12's "does not write" clause but left the row's closing sentence, *"C12 is terminal on
the frozen checkpoint,"* untouched. That is inconsistent with `CLAIMS.md`'s own status vocabulary, which
defines a terminal row as one where *"no further evidence will be sought … not a live loose end."* The
`U_info` write test is a **live loose end and is runnable on this very checkpoint**: `S_ov` is
`anchor_gates.suff_recovery` projected on C16's `U_info` against a covariance-matched null, with the
`PP-ovplant` sensitivity gate — frozen-checkpoint patching on the existing cohort, needing no retraining
and no new data. C12's status is therefore **bounded with one arm open**, not closed, and the B1 A-mixed
branch — *"the carrier writes a carrier-specific component onto the donor readout subspace that is
behaviourally redundant/masked for coupling"* — stands as an explicit open question.
**Registered as open in:** `CLAIMS.md:35` (status and closing sentence), `docs/ROADMAP.md` open problem 5,
`MECHANISM.md`, and the `preregistrations/README.md` B1 v2 row.
**Superseded in one respect by R36:** this addendum called the test "the cheapest unanswered question in the
record" and said it "should be run". **Both halves are wrong** — see R36. It is not runnable as scoped.

## R36 — B1's per-head OV endpoint is NON-IDENTIFIABLE on this checkpoint, not merely uncomputed (2026-07-25)

**What changed.** R31 recorded that B1's preregistered OV endpoint (`S_ov`, `Ω`) was never computed and
called the A-mixed branch "unexcluded". That is correct. Its addendum then went further and called the test
**runnable now on the frozen checkpoint** — and that is **wrong**. Preparing to run it produced three facts
that together make the per-head OV branch **unevaluable as preregistered**:

1. **The protocol scopes every per-head OV claim to L0 readers, for a stated methodological reason.**
   Family control: *"per-head OV claims restricted to **L0 readers** (L1/L2 readers' value-restore absorbs
   upstream global-P propagation, must-fix #7)"*. Every outcome branch inherits it — A-OV requires "≥1 **L0**
   head", A-null(donor) requires "no **L0** head's `S_ov^h`", A-mixed requires "an **L0** `S_ov^h` beats the
   null". The restriction is the red-team's fix for the confound the protocol itself names: `V = W_V·r`
   depends on the position-carrying residual, so `Mv_h` is generically nonzero via propagation even for a
   functionally routing-only head.
2. **No L0 reader exists.** Preflight R scanned **76 heads including 9 at L0; not one reached Level A** at
   `BAND_P10` (best L0 `m_T` = 0.591, `significant: false`). `R_1024` = {L2×2, L3×6, L4×1} — **zero L0
   heads** (`results/phase-lattice/carrier_preflight_r.json`).
3. **The deferral gate cannot fire in the branch it exists to exclude.** `run_carrier_ov_av.py:24-26` defers
   the covariance-matched null and `PP-ovplant` to a follow-up *"**ONLY** for heads whose core value main
   fires (CI-low > SESOI_v)"*. A-mixed is *by definition* coupling-null with a positive `S_ov`. A
   coupling-null head never fires that trigger, so `S_ov` is never computed in exactly the case A-mixed
   describes. The promised `--null` follow-up mode was **never implemented** — no such flag exists.

**Rule.** Running `S_ov` on `R_1024` as it stands would compute the confounded deep-reader quantity that
must-fix #7 removed **on purpose**; it would not close A-mixed, and reporting it would manufacture a new
over-claim of exactly the kind this ledger exists to prevent. The correct disposition is the protocol's own
**Endpoint B — unresolved: "C12(donor-OV) unreached on this evidence."** The A-mixed branch is **not
excluded and not excludable on this checkpoint as scoped**. Closing it requires either (a) an
owner-authorized, logged amendment extending per-head OV claims to deep readers — which re-admits the
global-P propagation confound and therefore weakens what any positive could mean — or (b) a checkpoint or
reader set in which an L0 head reaches Level A at `BAND_P10`. Neither is a re-run of the existing design.
**No compute was spent:** this was settled by reading the frozen protocol against `carrier_preflight_r.json`.
**Sites corrected:** `CLAIMS.md:35`, `docs/ROADMAP.md` open problem 5, `MECHANISM.md`,
`preregistrations/README.md` B1 v2 row, and R31's addendum above.

## R32 — C41 was labelled "confirmatory" in the ledger while R28 records it at verification tier (2026-07-25)

**What changed.** `CLAIMS.md:64`'s evidence cell read *"…**GENERIC READOUT AXIS**, confirmatory"*, and
`G1_REPORT.md:24`, `REPRODUCIBILITY.md:154` and the preregistration index agreed. **R28 — filed the previous
day — already records that G1's frozen §9 `Global PASS = AND(...)` was never completed**, so the ledger and
its own correction record contradicted each other. Re-verified in code and artifact: the **planted-confound
arm was not run at all** (`match_pairs(..., extreme=True)` is unreachable in
`run_g1_cross_example_transduction.py`), and **planted-positive was applied as a point check**
(`monotone ∧ slope > 0`, slope **0.785**) rather than the preregistered **slope-CI-covers-1** interval form.
**Rule:** C41 is reported at **verification tier, not confirmatory**. The *finding* is unchanged and is not
weakened — the C16/C26 final-residual delivered state transfers as a **generic readout axis**, not
example-specific information; the adversarial verification stands at `RESULT_STANDS_WITH_CAVEATS`. What
changes is the tier word only. "established" is retained, matching the house precedent at C34 ("established
on discovery; gate-qualified verification on confirm"). A confirmatory G1 requires the interval-form
planted-positive gate and the planted-confound arm.
**Sites corrected:** `CLAIMS.md:64`, `G1_REPORT.md:24`, `REPRODUCIBILITY.md:154`,
`preregistrations/README.md:44`, `docs/METHODOLOGY.md:39`. **Deliberately not scoped past R28:** the
`--smoke`/`D_src` cohort assignment for planted-positive is prereg-licensed (§4, §9) and is *not* a
deviation; calling it one would create a second false statement.
**Left as dated record:** `DECISIONS.md:201`. (A second dated site, the 2026-07-24 draft-sourcing audit, was removed with its file in the 2026-07-31 cleanup; its unapplied findings became R38 and R40.)
**Additional site corrected (2026-07-31, ledger restructure):** `CLAIMS.md` C38 carried a second,
missed instance — a clause describing C41 as *"confirmatory on the Atlas `H_conf` split"*, which
contradicted C41's own row and this entry. R32 corrected five sites on 2026-07-25 and this one was
overlooked; it surfaced during the ledger restructure. The false tier word is **removed rather than
paraphrased forward** (restating it in other words would create a second false statement), and C38 now
defers to C41's own row for the tier instead of asserting one. The finding is unchanged: the donor-side
delivered state still transfers as a generic readout axis. **Lesson:** correcting a claim's own row is
not enough — grep the ledger for every row that *describes* that claim, because a cross-reference can
carry a retracted status word forward invisibly.

## R33 — C39's sibling band-deletion is verification-level, and its artifact `verdict` is a hand-written adjudication (2026-07-25)

**What changed.** C39's claim cell said band-deletion *"confirms period-3 is **used coding-specifically**"*.
Two problems.

**(a) The decision rule was not met.** Only **12** energy-matched control bands qualified against a
requested **30**, so the add-one permutation p is floored at 1/13 = **0.077** — above the runner's own 0.05
threshold. `p_did = p_exon = 0.0769`. The arm cannot reject at its stated rule, so it is **verification
level, not a passed confirmation**. Clearing the rule would need ≥20 qualifying control bands (1/21 = 0.048).

**(b) The shipped `verdict` is not a machine output.**
`results/origin-siblings/canonical/sibling_carrier_bandtransfer.json` carries
`verdict: "USED_coding_specific"`, but `run_sibling_carrier_bandtransfer.py:222` can emit only
`VOID_unhealthy` (which it would, since `period3.healthy = false`) or, with the health channel corrected,
`NOT_USED` (since p = 0.077 > 0.05). **The value that shipped is not in the runner's range.** The machine
value is preserved beside it as `verdict_mechanical: "VOID_unhealthy"`, with a `verdict_note` giving the
reasoning. **State this fairly: the adjudication is transparent, reasoned, and preserved — it is not
concealed, and its scientific argument is defensible.** The note correctly identifies that the mechanical
gate used `min(exon, intron)` headroom when *exon headroom is the outcome* (the R13 error), and that the
corrected non-outcome health channel — intron prediction — retains 87.6%.
**What is nevertheless wrong is the governance:** the adjudication was written **into an immutable artifact**,
overwriting the machine field, so a re-run now disagrees with the committed file; it carried **no
correction number** until this entry; and **three surfaces disagreed on status** — the sibling manifest says
`"status": "EXPLORATORY"`, `results/MANIFEST.csv:756` says `canonical / Behaviour / positive`, and
`CLAIMS.md:62` said *established*.

**Rule:** report the band-deletion as **reproducing C8's exon-selective damage pattern at verification
level**, never as a confirmation. **The corpus-attribution conclusion of C39 is unchanged and remains
established** — it rests on the sibling P-spectrum contrast (72-sp 435× vs human-only 6.2×), not on this
arm. Going forward, runners emit `verdict_machine` and any human adjudication goes in a sibling
`*.adjudication.json`, **never** into the result. **Do not edit the three affected artifacts** —
`sibling_carrier_bandtransfer.json`, `a11_address_geometry.json` (machine `ABSOLUTE` preserved as
`verdict_raw_script`), and `superseded/a9_payload_semantics.json` all show this pattern and are immutable;
this entry is their disclosure. **Do not re-run the band transfer with more control bands to push p under
0.05** — the outcome has been seen, so that would be outcome-aware verification presented as confirmation.
**Sites corrected:** `CLAIMS.md:62`, `MECHANISM.md:120`,
`docs/CARRIER_ORIGIN_SYNTHESIS_2026-07-23.md:43`, `results/origin-siblings/RESULTS_INDEX.md:15`,
`REPRODUCIBILITY.md:164`.

## R34 — LOGGED AMENDMENT: commit c38e1ba's "additive" tier banners broke 17 content-addressed pins (2026-07-25)

**What changed.** Commit `c38e1ba` ("Tier-2 formalization: results reorg, repo-wide tier stamps…") inserted a
one-line tier banner into documents whose SHA-256 is **pinned by other artifacts**, and `git mv`-ed 229
development artifacts into `development/` subtrees. Its own message asserts *"Scientific record unchanged:
read-only ledgers/preregs touched only by additive tier stamps."* **An additive edit to a hash-sealed
document is a breaking edit.** Measured at HEAD, 17 pins are broken:

- **C36's frozen confirmation package** (`p3_phase_surgery_confirmation_package_v1.json`, 43 bound records):
  24 verify, 6 are gitignored model weights, and **13 fail** — 10 *relocated* (byte-identical at
  `results/phase-lattice/development/`, so moved but not modified) and **3 content-drift**:
  `PREREGISTRATION_P3_CLOSURE.md`, `REDTEAM_SYNTHESIS_P3_CLOSURE.md`, and
  `freeze_p3_phase_surgery_confirmation.py`. Its committed audit still records `"status": "PASS"`.
- **All four Head–Context-Atlas protocol pins** (`ANALYSIS_PLAN_V1.md`, `ANALYSIS_AMENDMENT_A1.md`,
  `MECHANISM_VALIDATION_AMENDMENT_A1.md`, `MECHANISM_VALIDATION_AMENDMENT_A2.md`), across **29 stored
  occurrences**. `build_atlas_mechanism_package.py:295` raises `RuntimeError("analysis-plan hash drift")` on
  these, so **the atlas mechanism package cannot currently be rebuilt**.

No test exercised either gate, so the suite stayed green throughout.

**Owner decision (2026-07-25): log the amendment; do NOT revert the banners.** The alternative — reverting
the two-line stamp on the seven affected files to restore byte-exact verification — was considered and
rejected in favour of tier-stamp uniformity.
**Rule:** the pre-stamp and post-stamp digest of every affected file is recorded in
`tests/data/hash_pin_ledger.json`, and `tests/test_hash_pin_ledger.py` asserts the broken set is **exactly**
that ledger — so new drift fails, a repaired pin fails (forcing the ledger to be pruned), and widening the
allowlist is a visible diff to a checked-in file. **No scientific content changed:** `git show c38e1ba`
shows the prereg diff is a single added banner line, and the ten relocated artifacts are byte-identical at
their new paths (asserted every run). The frozen package, its 2026-07-21 PASS audit, and every result
artifact are **unchanged and must stay unchanged** — this entry, not an edit, is their disclosure.
**Standing rule going forward:** before editing any file, grep the result trees for its basename near
`sha256`. "Additive" is not a safe category in this repository.

## R35 — the immutability rule was documentation only; a guard now enforces it, and the residual hazard is inventoried (2026-07-25)

**What changed.** `README.md:152` states *"no new run may silently overwrite the scientific record"* and
`REPRODUCE.md` §4 states *"Write a new immutable artifact; do not overwrite the controlling result."*
**Nothing enforced either.** Measured: **56 runners** default `--out` to a path listed as an artifact in
`results/MANIFEST.csv` — **41 of them `canonical`** — and `grep -rc "out).exists()" scripts/` returned
**zero** files. `ExperimentManifest.write` had no existence check. Reproducing any claim by its documented
command overwrote the artifact it was meant to be compared against.

**Rule.** A new module `src/gfmcd/artifact_guard.py` provides `ArtifactExistsError` and
`guard_output_path(path, *, overwrite=False)`. The parameter is **keyword-only** and defaults to refusing,
with no environment escape hatch. It **refuses rather than auto-suffixing**: an auto-suffixed sibling looks
like a successful rerun while quietly not being the artifact anyone cites. New runners call it before
writing anything.

**Why a new module, and what that costs.** The guard was first added directly to
`src/gfmcd/provenance.py`, so that `ExperimentManifest.write` would inherit it. That edit **broke a hash
seal**: `provenance.py` is one of the 19 code files bound by C36's frozen confirmation package, and
`tests/test_hash_pin_ledger.py` — written hours earlier in this same session — failed on the next run and
named the file. The guard caught its own author. Rather than widen the known-broken ledger to accommodate a
**new and entirely avoidable** drift (the c38e1ba entries above are unintended historical drift; this would
have been deliberate), the change was reverted, `provenance.py` restored byte-identical to its pinned
digest `05d3cd48…`, and the guard moved to an unpinned module.
**The cost, stated plainly:** `ExperimentManifest.write` is therefore **not** guarded. That is a smaller
loss than it sounds — `write` persists *manifests*, whereas the hazard is the *result* JSONs, which the 56
runners write by their own means and which never passed through it. The real protection is the inventory
assertion below. Guarding `write` remains available whenever someone is willing to spend the C36 code pin
on it; that is an owner decision, not a mechanical one.

**Why the 56 runners were NOT edited.** They write their *result* JSON by their own means, and their
producing commits are **pinned by the manifests beside those results** — editing them to insert a guard
call would desynchronise a pinned provenance chain in order to fix a documentation defect, which is the
larger harm. They are therefore **inventoried, not edited**:
`tests/data/unguarded_out_defaults.json` records all 56 with their `--out` default and manifest status, and
`tests/test_overwrite_guard.py` recomputes the set by AST on every run and asserts it **has not grown** —
so a new runner that defaults `--out` at the record fails CI the day it lands, while the existing 56 stay
byte-stable. Fix forward in successor runners. Until then, **always pass an explicit `--out` to a new path
when reproducing a claim** (documented at `REPRODUCE.md` §2.2).

## R37 — the "≈10.63 carrier is fitted to ±0.005" precision claim withdrawn (drafted 2026-07-24; renumbered on merge 2026-07-30)

> **Numbering note.** This correction was drafted as **R29** on 2026-07-24 but was never merged. The `master` line independently assigned **R29** to the L2H15 unit error on 2026-07-25, so the two collided. The published R29 keeps its number (it is already cited from `RESULTS_VERIFICATION.md`, `docs/PREREG_CONFORMANCE.md`, `run_l2_accounting.py` and `run_a3_l2h15_address_payload.py`) and this entry is renumbered to the next free slot. Nothing in its content changed. Item **(b)** below — that the nucleosomal comparator is mouse and the docs never said whose DNA — was subsequently **resolved by R30**; it is left as written to preserve what was open at the drafting date.

**What was claimed.** Five live documents stated that the ≈10.63 carrier "is fitted to ±0.005" nt, using
that figure as the uncertainty against which the 0.33-nt gap to the ≈10.298 nucleosomal signal was called
a powered refutation.

**Why it was wrong.** No artifact supports ±0.005 as a fit uncertainty on the carrier.
`results/phase-lattice/carrier_subbin.json` records **two** multi-species sub-bin fits — **10.6074**
(stage-1, 512-row grid, bin 48, r²=0.9996) and **10.6524** (final, 1026-row grid, bin 96, r²=0.9405) —
whose **spread is 0.0450 nt**, nine times the claimed bar; the reported 10.630 is their mean
(`consensus_wavelength` = 10.62994). `run_carrier_subbin.py` itself only certifies the two grids as
`CONSISTENT` under a **0.05-nt** tolerance (`d < 0.05`), which 0.045 barely clears. The estimator's own
detuning control at bin 48 gives scatter **0.0955 bins = 0.0212 nt** (bias 0.0022 nt) — still 4× the
claimed figure. The literal `0.005` in the artifact is
`control.abs_tone|10.7789|d0.76|L512|s0 → err = 0.005`, an error in **bin units** on a synthetic tone at
true bin 47.5, i.e. **0.0011 nt** — a plumbing check that the estimator recovers a known synthetic input.
Two compounded slips: **bin units read as nt**, and a **synthetic control read as a measurement
uncertainty**.

**What is unaffected.** C14 stands, and the refutation is if anything better characterised. The gap is
**+0.332 nt** = 7.4× the carrier's two-grid spread (0.045) and 15.6× the estimator scatter (0.021); the
nucleosomal signal's own three estimators span **0.0451 nt** (fine-scan 10.2976, genic 10.2959,
phase-rotation 10.2525), also far inside the gap. No result artifact was edited and no number in
`carrier_subbin.json`, `dyad_real.json`, or `dyad_wavelength.json` changed.

**Rule.** State the distinctness as the gap **relative to the measured spreads** — *carrier 10.63 ± 0.05,
nucleosomal 10.30 ± 0.05, gap ~7× the larger uncertainty* — and never quote a ±0.005 carrier fit
precision. `docs/NOMENCLATURE.md` §2 carries the guard. Corrected in `MECHANISM.md`,
`RESULTS_VERIFICATION.md`, `docs/NOMENCLATURE.md`, `docs/BIOLOGICAL_BRIDGES.md`, and
`figures/splicebert/fig4_carrier_circuit.CAPTION.md`. **Not** corrected, by frozen-record convention:
`PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR.md:73` and the dated T2 review/handoff docs
(`docs/REVIEW_2026-07-22.md:44,294`, `docs/OPEN_TRACKS_EXPANDED_2026-07-22.md:64`,
`docs/TIER2_HANDOFF_2026-07-23.md:252`) — these are historical records of what was believed at their date.
(The last two of those sites were removed with their files in the 2026-07-31 record cleanup; the list is
appended to rather than edited, since it records what was deliberately left uncorrected at its date.)

**Also noted (not corrected here; owner ruling needed).** (a) `results/phase-lattice/dyad_real.json`
anchors its matched filter on `T0 = 10.6667` — the **retired** grid label — as the reference wavelength;
the reported 10.298 comes from the fine scan, so no conclusion depends on it, but a retired label is
load-bearing inside a canonical result file. (b) The nucleosomal signal is **mouse** (mm9 ESC, Voong 2016
GSM2183909) while the carrier is vertebrate/human; "measured in DNA" in `docs/NOMENCLATURE.md:27` and the
fig4 caption never says whose DNA. Helical periodicity is not species-specific at this resolution, so the
distinctness argument is unaffected.

## R38 — the human-only "best head has `m_T = 0.001`" was L1H3's band mass, not an effect size (2026-07-31)

**What was claimed.** `figures/splicebert/EVIDENCE.md` § "Why the human-only null is a null and not a
blind instrument" answered the lone-reader power objection with *"the best human-only head has
`m_T = 0.001`"* — i.e. the null is safe because the strongest human-only effect is ~0.

**Why it was wrong.** `results/phase-lattice/fig_report_data.json` (`levelA.human_only`, 75 heads)
records the best human-only head as **L2H9 at `m_T = 0.556`**, with **`p_exact = 0.796`** — 318 of its
400 permutations match or exceed it. **No human-only head has `m_T = 0.001`**; that figure is L1H3's
**band mass** (`0.00193`, the 0.19% quoted correctly two paragraphs earlier in the same section), a
different quantity read as an effect size. Found by the 2026-07-24 draft-sourcing audit, which recorded
the discrepancy but did not edit the of-record document.

**What is unaffected.** **C13 stands, and the argument is stronger on the correct number:** the closure
holds because the best head's apparent signal sits **inside its own permutation null**, not because the
effects are ~0 — a stricter refutation of the blind-instrument objection than the one written. Zero of
the 14 preregistered readers reach Level A after add-one correction (R19), the twin-model sensitivity
argument (eight Level-A heads multi-species) is untouched, and no artifact value changed.

**Rule.** `m_T` and band mass are separate quantities on separate scales — quote each with the head it
belongs to. A power-objection rebuttal must report the best head's `p_exact` against its own null, not a
bare effect size. Corrected in `figures/splicebert/EVIDENCE.md`.

## R39 — four period-3 memorial test files were deleted, and the missing-artifact correction they were waiting for was never written (2026-07-31)

**What changed.** `tests/test_carrier_phase_memorials.py`, `test_carrier_strata_memorials.py`,
`test_dual_reader_memorials.py`, and `test_dyad_assay_memorials.py` were removed from the tree. Each was
gated on an artifact that no longer exists — `results/phase-lattice/{carrier_phase,carrier_strata,dual_reader,dyad_assay}.json` —
and on a runner that no longer exists, both deleted in `af64808`. Every test in all four files had been
skipping since that commit and could never fire again; `results/MANIFEST.csv` carries no row for any of
the four artifacts.

**Why this needs an entry rather than a silent delete.** The artifact deletion in `af64808` was itself
never logged. `docs/PREREG_CONFORMANCE.md` already flags the gap for the carrier-phase family — the
remediation it names ("delete the test *or* restore the artifact, and log the correction") was half
applied, twice: the artifacts went, then the tests went, and neither step wrote a correction. This entry
closes that. **No claim is affected:** none of the four artifacts is cited by any live claim row, and the
period-3 record rests on `carrier_surgery.json`, `carrier_dose.json`, `carrier_mediation.json`, and the
C36 confirmation package, all present and all still covered by memorial tests.

**What was lost.** The four suites' *assertions* are gone with them. Whatever invariants they encoded
about the carrier-phase, carrier-strata, dual-reader, and dyad-assay outputs are recoverable only from
Git history, and if any of those experiments is revived its memorials must be rewritten from the
preregistration rather than restored from the deleted file — a memorial written against an artifact that
no longer exists cannot be assumed still valid.

**Rule.** Deleting a result artifact and deleting the memorial that guards it are two separate edits to
the scientific record, and **both** require a correction entry. A permanently-skipping test is not
harmless: it reads as coverage in the suite count while asserting nothing.

**One deletion in the same pass was reversed for this reason.** `src/gfmcd/circuits/attribution.py` was
removed as dead code — verified zero importers, and its only test was itself permanently skipped on a
`colab_results/` path deleted in `af64808`. But the **frozen** `PREREGISTRATION_B4_L3H1_CARRIER_TO_L2MLP.md:163`
cites it by path in an inventory of the ACDC-family path/edge-patching primitives the repository has, and
B4 ran (22 result rows). Deleting it would have left a frozen protocol pointing at nothing and quietly
weakened a documented capability claim, so the module was restored. Its permanently-skipped test was not.
**Rule:** "zero importers" is not sufficient grounds to delete a module — grep the frozen preregistrations
and red-team syntheses for its path first, because those cite tooling as evidence and cannot be edited.

## R40 — C41 rendered across-draw ranges in confidence-interval notation, and quoted a pilot dose slope as a confirmatory one (2026-07-31)

**What was claimed.** C41's row wrote `Δ_spec = −0.33 [−0.345, −0.276]` and `τ_PERP = 0.00 [−0.015, +0.010]`
in the same bracket notation every other row in this ledger uses for a confidence interval, and gave the
dose slope as `β = 0.27`.

**Why it was wrong.** Verified directly against `results/splicebert_coupling/g1_cross_example_transduction.json`:

- Those brackets are the **min/max across the 8 resampled-source draws** (`across_draw.Delta_spec`,
  `across_draw.PERP_MATCHED_lin`), not intervals. Across-draw mean `Δ_spec` is **−0.320** (median −0.330,
  SD 0.023); `τ_PERP` is **−0.0001**. The **per-draw dyadic CIs are far wider** — draw 0's `Δ_spec` CI is
  `[−0.527, −0.220]`, half-width **0.153**, roughly 4× the full across-draw spread. A reader taking the
  notation at face value understates the uncertainty by that factor.
- `β = 0.27` is not the confirmatory value. The confirmatory across-draw mean of `beta_dose_REAL` over the
  8 draws is **0.2425** (median 0.240, range 0.215–0.266). `0.27` matches the `H_dev` **pilot** (0.2737) —
  a pilot number quoted in a confirmatory summary.

**What is unaffected.** **The finding is unchanged and none of the verdicts move.** Δ_spec is negative in
all 8 draws and its across-draw spread does not approach zero; τ_PERP straddles zero in every draw; the
modal verdict stays 6/8 GENERIC, 0/8 TRANSDUCING. The dose relation is still monotone and positive — only
its slope was overstated by about 11%. C41's tier is untouched (verification, R28/R32).

**How it was found.** The 2026-07-24 draft-sourcing audit recorded both, but the document was a dated
review and no one applied its §6; the 2026-07-31 cleanup deleted it, and the finding surfaced only because
a verification pass diffed the deleted file against what had actually been salvaged.

**Rule.** In a claim row, `[a, b]` means a confidence interval and nothing else. A stability spread across
seeds or draws must be written as such — "range", "SD", or "min–max across N draws" — because the two
carry opposite messages about precision. And a pilot value may never appear in a confirmatory summary
without being labelled: check which split produced every number before it enters a row.

## R41 — reproducibility gaps found by a dedicated audit: a canonical figure producer could not run, and twelve live memorials had been silently dead since af64808 (2026-07-31)

**What was wrong.** Three defects, all pre-dating the 2026-07-31 cleanup, all invisible to the test
suite because the guards check paths and IDs, never whether a runner can execute.

1. **C8's figure-data producer could not run.** `scripts/splicebert/make_fig_phase_data.py` — the
   canonical producer of `results/phase-lattice/fig_phase_data.json` — does an unguarded module-level
   `from make_fig_wave_fit import build`, and `make_fig_wave_fit.py` was deleted in `af64808`. Any
   attempt to regenerate that artifact died at import. **Fixed:** the module (102 lines, a pure
   function of already-recorded `R(k)`, depending only on `src/gfmcd/splicing/wave.py`, which
   survives) was restored from `af64808^`.
2. **Twelve memorials for a live claim had been dead since the same commit.**
   `tests/test_chromatin_refutation_memorials.py` pins R17 — the powered refutation of the nucleosome
   reading behind **C14**. Its module-level `skipif` ANDed three artifacts; one of them,
   `results/phase-lattice/synth_grid.json`, was deleted in `af64808`, so **all twelve tripwires
   skipped**, including the seven that need only `dyad_wavelength.json` and `dyad_real.json` — both
   present the whole time. Twelve memorials defending a live refutation counted as suite coverage
   while asserting nothing. **Fixed:** the gate is now per-artifact; seven tripwires run again. The
   five D27 grid tripwires remain skipped and are now *individually* marked with the reason —
   `synth_grid.json` is gone and no runner in the tree regenerates it, so that leg of R17 is
   **unverifiable from this checkout**.
3. **Several canonical runners hardcode absolute paths from the original machine.** 21 tracked files
   carry a non-CLI-overridable `/Users/cberg/...` constant. Most are `DEFAULT_*` values feeding an
   argparse default and can be overridden; the following cannot without editing source:
   `measure_band_mass.py:46`, `run_postsoftmax.py:380`, `run_dyad_real.py:57-58`,
   `run_dyad_wavelength.py:120`, `run_splice_phasing.py:64-65`, `audit_ed1_provenance.py:56-58`,
   `src/gfmcd/splicing/circuit.py:73-74`, and — worst — `run_g1_cross_example_transduction.py:94-95`,
   the **C41** runner, which hardcodes not only the genome but the *repository checkout path itself*,
   so it does not run from a clone or a worktree. **Not fixed here:** these runners are pinned by
   their manifests and R35 records the standing decision not to edit them, because changing a default
   changes what a rerun does. Disclosed instead in `REPRODUCE.md` §2.2 item 5.

**What is unaffected.** No claim, number, interval, or evidence level changes. Every artifact these
runners produced is committed and its recorded values stand; what was broken is the ability to
*regenerate* them, and — for the twelve memorials — the ability to notice if a committed value drifted.

**Rule.** "The suite is green" and "the record is reproducible" are different properties, and this
repository's guards only measure the first. A memorial gated on an artifact that no longer exists is
worse than no memorial: it occupies a slot in the coverage count. When an artifact is deleted, grep the
test suite for its path and split any conjunctive skip gate that mentions it, so the tests that do not
depend on it keep firing.


## R42 — HCA5 and HCA8 removed from the Atlas ledger: their controlling artifacts were never committed and are not obtainable (2026-08-01)

**What changed.** Two rows were removed from `studies/splicebert_head_context_atlas/CLAIMS.md`. This is a
deliberate exception to the never-delete rule, taken by the repository owner, and the rows are preserved
verbatim below so nothing is erased — only relocated out of the live ledger.

**Why.** Both cited `head_effect_summary.json` and/or `head_effect_matrix.jsonl`. Neither file exists
anywhere in the tree and neither has ever been committed on this branch
(`git log --all -- '*head_effect_matrix.jsonl'` is empty), so neither claim was checkable by a reviewer.

- **HCA5** was already terminal: *closed (competition branch; **withdrawn / invalid estimand**)*. The
  rank-concentration statistic it reported is not faithfulness, and its pooled acceptor rows used the
  wrong coordinate. It had been superseded by HCA10's exact joint donor-only curves. Removing it costs
  the record nothing that HCA10 does not already carry.
- **HCA8** is the narrower case and worth stating precisely: **one of its two artifacts does exist.**
  `head_effect_donor_valid_reanalysis_v1.json` is committed and canonical; only `head_effect_matrix.jsonl`
  is missing. Its surviving, checkable content — that competitor-destroy and matched-decoy-destroy recruit
  largely the same head family, so no distinct competition module is licensed — is **already carried by
  HCA10**, which reaches the same conclusion on exact joint patches and whose artifacts are all present.
  **HCA9** remains the sole live consumer of `head_effect_donor_valid_reanalysis_v1.json` and is
  unaffected; it is the salvaged L3H1 positive control cited from the donor-circuit ledger.

**Downstream edits made in the same change.** Three `results/MANIFEST.csv` / `MANIFEST.json` rows carried
composite claim IDs naming the removed rows and were repointed so every ID still resolves:
`HCA5/HCA7-HCA10` → `HCA7-HCA10`, and `HCA8/HCA9` → `HCA9` (two rows). `scripts/README.md` and the study's
`RESULTS.md` were repointed identically, and `RESULTS.md`'s `head_screen_v1` row is marked retired.
**No artifact was deleted** and no number in any surviving row changed.

**The removed rows, verbatim:**

```
| HCA5 | Single-head concentration identifies whether C is sparse or distributed | Mediation | closed (competition branch; **withdrawn / invalid estimand**) | historical train+val screen | `head_effect_summary.json`, `AMENDMENT_HEAD_SCREEN.md` | The rank-concentration statistic is not faithfulness, and pooled acceptor rows used the wrong coordinate. Replaced by exact joint donor-only curves (HCA10). |

| HCA8 | On coordinate-valid donor targets, competitor-destroy C and matched-decoy-destroy D recruit largely the same head family (`r=0.804` signed; `r=0.942` magnitude). Per-head C>D magnitude tendencies exist, but do not establish a distinct competition module. | Mediation | closed (competition branch; supported exploratory, donor-only) | train donor targets, intron-mate ≤1 kb; chr8/9 untouched | `head_effect_donor_valid_reanalysis_v1.json`, `head_effect_matrix.jsonl` | Pooled/acceptor numbers are withdrawn. Exact joint closure in HCA10 resolves the nonadditivity question; behavioral specificity remains failed. |
```

**Rule.** A claim whose controlling artifact was never committed is not a weak claim — it is an
unverifiable one, and it should not sit in a ledger a reviewer is told is the point of contact. Removing
it is legitimate *only* when the row is preserved in this file and every citation is repointed in the
same change, so the record loses nothing but the live ledger stops asserting what it cannot support.

## R43 — C40's published `n_sig` was not governed by the seed its manifest records, and the overwrite guard was blind to the idiom that dominates the two newest studies (2026-08-01)

Two independent defects found by a reproducibility audit. Both are pre-existing; both were verified in
code before this entry was written.

**1. `n_sig` is seeded from CPython's randomized `hash()`, while `seed: 0` is recorded.**
`scripts/crossmodel/run_reader_freqcontrol.py` drew its per-head permutation RNG from
`np.random.default_rng(hash((h, round(T, 4))) % 2**31)`, where `h` is a head label string such as
`"L3H1"`. CPython randomizes `hash()` of a `str` per process unless `PYTHONHASHSEED` is set — verified:
three runs of `python3 -c "print(hash(('L3H1',0.1)))"` give three different integers, while the same
call under `PYTHONHASHSEED=0` is stable. **The runner does not set `PYTHONHASHSEED` and neither the
artifact nor the manifest records it**, yet both declare `"seed": 0`. That declaration therefore does
not govern this statistic. C40's row quotes the affected numbers verbatim — *"n_sig 14 < frequency-control
grid max 28; period-3 n_sig=0"*.

**Status of the claim.** The seeding is fixed going forward: the draw now uses the repository's existing
stable-digest idiom (`shash`, `run_ic1_distal_intron_necessity.py:48`), so a rerun is deterministic
regardless of `PYTHONHASHSEED`. **The committed artifact was produced by the old code and cannot be
regenerated bit-identically** — not by the new code, and not by the old code either, which is the actual
problem. C40's substantive conclusion is not knife-edge (14 against a control-grid max of 28, with
period-3 at 0), so it is **not** withdrawn; but the exact `n_sig` integers should be read as one draw
from an unrecorded seed, not as a reproducible constant. Re-running C40 in the `gfmcd-mm` side
environment would settle it.

**2. The overwrite guard resolved only bare string literals.**
`tests/test_overwrite_guard.py::_out_default` returned a value only for
`ast.Constant` defaults, so it was blind to `default=str(RESULTS_DIR / "name.json")` — the idiom the
Atlas and GTEx trees use. Consequence: `analyze_b5_context_signal_power.py` defaulted `--out` to
`results/splicebert_head_context_atlas/b5_context_signal_power_v1.json`, which is the **canonical
controlling artifact for the live claim HCA23**, with **no exists-check and no guard call**, and it was
absent from the 56-entry inventory that exists to catch exactly this. `scripts/verify_manifest.py`
passes regardless — it checks that listed paths resolve, never that a runner might overwrite them.

**Fixed.** The resolver now handles `str(...)`, `NAME / "literal"` chains, and module-level path
constants including the `Path(__file__).resolve().parents[N]` root anchor. `analyze_b5_context_signal_power.py`
now calls `guard_output_path` before writing — it is a pure analysis script with no pinned-provenance
argument, so **R35's "inventoried, not edited" exemption does not cover it**. The inventory is
re-baselined 56 → 63; the seven additions carry `detected_by` and were invisible to the old checker,
not newly created.

**Rule.** A guard that silently fails to parse is worse than no guard, because the inventory it produces
reads as complete. When a checker walks source, it must be tested against the *idioms actually in the
tree*, not only the one the first author happened to use — and any statistic seeded from `hash()` of a
`str` is not reproducible unless `PYTHONHASHSEED` is both set and recorded.
## R44 — corruption alts constrain what a substitution WRITES, not what it CREATES; the donor-side operator plants a splice motif wherever the readout leaves it visible (2026-08-02)

**Outcome in one line: a real instrument defect, found and fixed, that changed no claim.** Every
donor-side arm exposed to it was re-run with a motif-clean alt and re-compared on an identical cohort;
**no row in any of the three ledgers moves.** Where anything shifted, the plant had been *suppressing*
the measured effect, not inflating it — the atlas's ranked-vs-random separation widens from 1.15x to
1.63x once it is removed. One gate came back **mixed** rather than clean and is reported as such below
(GFS25 V1: 3/5 seeds, not the >=4/5 its acceptance rule required).

**What was believed.** `ACCEPTOR_ALTS = ("CC","TT","CA","TC")` and `DONOR_ALTS = ("CC","AA","TA","AC")`
(`src/gfmcd/splicing/coupling.py:55-56`) were documented as substitutions "guaranteed to DESTROY the
motif and never create the other one", and `CORRUPT_ALT = ACCEPTOR_ALTS[0] = "CC"`
(`src/gfmcd/splicing/circuit.py:66`) was used by every confirmatory and mechanistic runner in the donor
arc. No reason to prefer `CC` within its tuple was ever recorded, and **no per-alt effect appeared
anywhere under `results/` before 2026-08-02**.

**The gap.** Writing `XY` at `[q-2, q)` touches three overlapping registers — `seq[q-3]+X`, `XY`, and
`Y+seq[q]`. The tuples constrain only the middle one, so a substitution can destroy its target and
simultaneously **create** a splice motif one base away. `tests/test_motif_cleanliness.py` proves
exhaustively over all 16 flanking contexts that cleanliness forces `X="A"` and `Y∈{C,T}`, leaving exactly
`AC` and `AT`; `MOTIF_CLEAN_ALT = "AC"` is clean at both sites.

**Measured, model-free.** Acceptor side: `seq[a-3]="G"` in **0.36%** (train, n=15,196) / **0.35%** (val,
n=5,930) / **0.5%** (GTEx cohorts, n=416) — `CC` plants a `GC` donor at a negligible rate; the acceptor
outlier is `CA`, which plants a rescued `AG` one base downstream in ~51–55% of exons. Donor side:
`seq[d-1]="G"` in **81.0%** (val GENCODE), **82.0%** (GTEx union), **87.8%** (atlas condition-B donor
sites) — `CC` plants a `GC` donor one base upstream, `TA` plants a **canonical `GT`** (95.5% on atlas
condition-B), `AA` plants an `AG` (62.0%), and `AC` plants nothing (**0.0%**) everywhere measured.
Artifacts: `results/splicebert_gtex_finetune_shift/gtex_planted_motif_census_v1.json`,
`results/splicebert_head_context_atlas/atlas_planted_motif_census_v1.json`.

**Exposure is decided by the READOUT, not by the alt.** Where the substituted index is **masked** the
base is invisible and nothing is planted; where it is **visible** the confound is live.

**1. `splicebert_partner_selection` — NOT EXPOSED; no claim changes.** 0 of 42 rows rests on donor-side
substitution; every corrupting row goes through `CouplingTask.pair` (`circuit.py:138-144`),
acceptor-side, at a masked-donor readout. `AG→CC` is **retained**, and is independently validated
alt-invariant within ±25% by **C42**. Swapping to `AC` here would *reduce* every effect ~11.6%
(AR1 `arms.ACC`, ρ = 0.884) and correct nothing. C1 and C2 carry an appended scope note; no number,
status word, or claim text changed.

**2. Head-Context Atlas — EXPOSED; re-run with `AC`; NO CONCLUSION CHANGES.** Largest measured plant
rate. Four arms re-run to new paths (originals byte-identical):

| arm | claims | committed → clean-alt | agreement |
|---|---|---|---|
| `run_all_head_screen` cond. B | HCA12, HCA18 | +0.1551 → +0.1573 (+1.4%) | per-head **r = 0.9986**; **L3H1 still rank 1**; top5 5/5, top20 20/20 |
| `group_mlp_closure` cond. B | HCA9, HCA10 | 0.1015 → 0.0976 (−3.9%) | r = 0.965; top10 9/10; L3H1-alone 0.0857 → 0.0810 |
| `behavioral_pilot` | HCA2, HCA3, HCA7 | 0.1748 → 0.1604 (−8.2%) | `M_comp` +0.0063 → −0.0125, both ≪ SESOI 0.05 — null holds; `sham` unchanged to machine precision |
| `specificity_pilot` | HCA4, HCA7 | −0.0473 → −0.0480 | specificity failure unchanged |

**The confound was suppressing the atlas, not inflating it**: in `group_mlp_closure` the ranked-vs-random
separation **widens** `1.15× → 1.63×` once the plant is removed, and `behavioral_pilot`'s MDE improves
`0.0241 → 0.0218`. Conditions C and D of the head screen reshuffle more (per-head r = 0.846 / 0.889,
top20 11/20 and 14/20), but their effects are ~0.02 against condition B's 0.155 — ranking 96 heads on an
effect that small is noise-dominated, and neither carries a claim that depends on head order; C is
already recorded as a non-specific "competitor-site perturbation response". **Only the two screens and
`group_mlp_closure` are pure like-for-like** (`variants[0]` on both sides); the two pilots also changed
from a 4-alt mean to a single construct, so their deltas mix the operator change with an averaging
change. Reported, not hidden.

**3. `splicebert_gtex_finetune_shift` — EXPOSED on CONTROLS; both follow-ups now CLOSED.** 8 of 34 rows
touch a donor-side destroy; in all seven runners the donor-side quantity is a positive control / health
gate, never the estimand — but GFS7, GFS11 and GFS24 quote its magnitude verbatim. The five published
`SpliceBERT.SpliceAI-GTEx.<seed>.pt` checkpoints (seeds 2020–2024) were obtained and verified against
the archive and per-seed sha256s already frozen in `configs/splicebert_gtex_finetune_shift/base.json`,
which unblocked both:

- **Classifier preflight, motif-clean third arm** (`native_classifier_preflight_clean_alt_v1.json`,
  51/51 checks pass). The committed `CC` and `AA` arms reproduce **bit-exactly** (delta 0.00e+00 on all
  five seeds), so only the added arm is new. Stratifying on `seq[d-1]=="G"` — the condition under which
  `CC` can plant — gives `AC-CC` of **-1.238** (95% CI -1.419 to -1.069, n=250) where a plant is
  possible and **-2.252** (95% CI -2.674 to -1.850, n=70) where it is not. The plant-attributable part
  is the **interaction, +1.015**, against a local destruction effect of 19.1–21.5, i.e. **~5%**. Note
  the raw difference is negative in *both* strata — writing `AC` is simply a milder edit at a donor than
  `CC`, independent of any motif — so quoting the G-stratum figure alone would overstate the confound by
  roughly 2x.
- **GFS25 V1 clean-alt recheck** (`positional_native_margin_gate_v1_clean_alt_recheck.json`). Identical
  replayed cohort, one changed input. `period3__delete` retention rises on every seed, by +0.007 to
  +0.041, and **3 of 5 seeds remain below the 0.80 floor**; `status` stays
  `STOPPED_INTERVENTION_PREFLIGHT` and `preflight.passed` stays `false`. The acceptance rule required
  **>=4/5** to say the stop reason stands unaided, so this is recorded as **MIXED**: the interaction the
  hypothesis predicted is real and measurable, but roughly an order of magnitude too small to explain
  the stop — seeds 2022 and 2024 needed +0.15 and +0.08 and received +0.007 and +0.041. Only seed 2021
  crosses, from 0.7835 to 0.8101, and it was already 0.0165 from the floor. **This settles only why V1
  stopped and cannot promote GFS25**, whose licensed claim rests on the V2 run that passed.

**A trap in the record, for anyone repeating this.** `native_classifier_preflight_v1.json` is a
**FAILED** preflight (`status: FAIL`, 31/51 checks, every seed `PASS=false`, clean donor margin ≈ -17,
`local_cc_drop_mean` *negative*). The passing run is **v2**, and it is v2 that carries the
`local_cc_drop_mean` 21.13 / five-seed range 19.1–21.5 figures. A clean-alt comparison made against v1
reads as a 21-logit shift where the true shift is zero.

**Gate fix.** `gate_no_planted_motif` (`scripts/splicebert_head_context_atlas/run_all_head_screen.py`)
rejects a "destruction" that creates a motif. It is kept **separate** from `gate_destroy_real_bases` —
which checks length, footprint and alphabet, and so could never have caught this — leaving that
function's verdict, and every artifact it gated, byte-reproducible. The planted-motif count is **always**
recorded into the run manifest; it is fatal only under `--require-motif-clean`. The live gate
independently reproduced the model-free census to the tenth of a percent (condition B 197/400 = 49.25%
recorded vs 49.2% predicted; C 34.5% vs 34.5%; D 33.5% vs 33.5%), and the clean-alt runs recorded zero
plants.

**No existing result is shown to be wrong.** The difference-in-differences probe built to detect the
planting in AR1's `DONA` arm came back **REFUTED** (`+0.014 [−0.022, +0.048]` against a ±0.265 bound),
and every atlas re-run above preserves its conclusion. `AC` removes a confound that would otherwise have
to be defended; it does not overturn anything.

**Two reproducibility gaps found in passing — recorded, deliberately NOT fixed here.** (a) The atlas
discovery config `studies/splicebert_head_context_atlas/colab_package/config.discovery.json` was removed
in the 2026-07-31 cleanup (`7e254fc`) although **every committed HCA manifest cites it**; it was
recovered from Git for these runs and its pinned GTF sha256 verified (`83fba3e9…13a453`), but as the tree
stands the atlas cannot be re-run from a clean checkout. (b) `run_group_mlp_closure.py`'s committed
invocation requires `/Users/cberg/Downloads/head_effect_reanalysis.json`, a machine-specific path outside
the repository (FAILURE_MODES C5). Re-adding a deliberately-removed file is a decision, not a cleanup,
so both are logged for the owner rather than silently repaired.

**Owner decision (2026-08-02): the GTEx donor-side runners were switched to `MOTIF_CLEAN_DONOR_ALT`.**
R44 originally left the seven runners unedited because each produced an immutable artifact. The owner
elected to switch them so a future re-run is clean by default. **The consequence is recorded here rather
than hidden: these seven runners no longer reproduce the committed artifacts they produced.** Any re-run
now measures the motif-clean operator, and a byte-comparison against `native_classifier_preflight_v1/v2`,
`native_mechanism_confirmation_v1`, `native_context_preflight_v1`,
`native_biological_features_preflight_v1`, `positional_band_surgery_v1` or
`positional_native_margin_gate_v1/v2` will differ for that reason and no other. The **acceptor**-side
substitutions in those same files are deliberately untouched — `CC` is validated there by C42, and the
census puts its plant rate at 0.5%.

`run_gtex_native_classifier_preflight.py` is handled differently: rather than replacing its `CC` arm it
**gains a third, motif-clean arm** (`local_ac_drop_mean`, `local_ac_drop_positive_fraction`,
`local_ac_minus_cc_mean`), because R44's open item 2 asks for an *estimate* of the attenuation rather
than the `+0.85` upper bound, and that needs `CC`, `AA` and `AC` measured on one cohort. The gate now
turns on the clean arm (`local_ac_sensitivity`); `local_cc_sensitivity` and `local_aa_sensitivity` are
retained as reported context, and the neutral-specificity floor now includes the clean arm so it cannot
be flattered by two contaminated ones. The `local_cc_*` keys keep their original meaning — they are
still the `CC` numbers — so no committed key changes meaning.

**Rule.** An alt tuple that constrains only the bases it writes is not a motif-safety guarantee, and a
gate that checks length, footprint and alphabet cannot catch what an edit creates. Any corruption whose
substituted index is VISIBLE to the model must use `MOTIF_CLEAN_ALT`, or measure and report its planted
rate.
