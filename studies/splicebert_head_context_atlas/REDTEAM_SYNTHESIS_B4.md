# B4 — L3H1/carrier → L2MLP bridge (HCA20 residual): 5-lens design red-team synthesis (2026-07-22)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Reviews:** `PREREGISTRATION_B4_L3H1_CARRIER_TO_L2MLP.md` v1 (DRAFT for pre-compute red-team). Five
adversarial lenses, run BEFORE any confirmatory compute (formalize-audit-verify steps 1–2). **Outcome: the
design is NOT freezable as the bridge study it names.** Two lenses returned `fatal` and a third lodged a
fatal-severity identifiability issue; all three converge on the **same** defect, independently verified against
code + data. This is an R23/R15-class pre-run catch: the cycle prevented an expensive run whose "positive"
would have been an uninterpretable confounded total-effect, and whose "null" is unreachable at budget.

**Disposition: DEFER (fold into B3).** Retain this prereg as design-of-record. The draft's own instinct
(DEFER / ride with B3) is correct, but its stated *primary* blocker (second-difference power wall + Drive-only
cohort) is **secondary and partly mis-stated**; the *primary* blocker is **architectural non-identifiability of
a direct carrier→L2MLP edge on a frozen post-LN checkpoint** — a defect no amount of Colab/CUDA compute, larger
n, or new Atlas freeze can fix.

| Lens | Verdict | Headline |
|---|---|---|
| Identifiability | **fatal** | reframed "carrier→L2MLP edge" is **not separable** from upstream carrier-reader mediation (L2H15 is a layer-2 comb head feeding L2MLP within the same block); no clean additive carrier component to subtract under post-LN |
| Confound completeness | **fatal** | any implementable carrier deletion at L2MLP's input is produced by a **globally carrier-deleted** forward → bundles the direct removal with **comb-collapse content routing** (L2H15/L0/L1); ΔΔ_bridge = direct + through-comb; controls B4.b/B4.c orthogonal to this backdoor |
| Calibration / intent | fixable (holed) | **no failable planted-positive travels the real channel** (PP-c8 wrong readout+locus; PP-bridge is a forced arithmetic injection); M-edge oversold as identifiability anchor when it is a plumbing/locality check; intent-drift (estimand has no L3H1 term) |
| Statistics / power | fixable | the "2–4× variance / P3.4 wall" analogy **mislabels a data-location problem as a power floor** (ΔΔ_bridge is a *paired* second difference → pairing typically *reduces* variance; HCA20 whole-B HW already 0.035 < SESOI); dyadic bootstrap mis-imported (one-way gboot exact); B4.b add-one floor under-powered |
| Test | pass | — |

---

## The convergent FATAL (three lenses, one defect; verified against code + data)

**Claim under test:** an *edge-restricted, direct* carrier → L2MLP-input path patch, distinct from
"C8 re-measured globally," whose ΔΔ_bridge second difference isolates whether L2MLP's partner write uses the
period-3 carrier.

**Why it is non-identifiable on this architecture — three independent verifications that all land on it:**

### F-1 — There is no edge-local carrier to delete; the only implementable deletion is global.
- The sole carrier-deletion primitive, `run_carrier_surgery.py::make` (L118–123), zeroes period-3 FFT bins on
  the **whole position-embedding weight matrix** `emb.weight` and rebuilds it (`np.fft.irfft`) — a **global
  edit at the source**, not a mid-layer edge hook. **Verified** (L108–123, `BAND=range(339,346)` L82).
- SpliceBERT is **post-LN**: each sub-block output is added to the residual and then LayerNormed
  (`circuit.py:555`, "write into the residual, BEFORE add+LayerNorm"). The running residual is renormalized at
  every sub-block, so there is **no per-source additive decomposition** to subtract a "carrier component" from
  at L2MLP's input (valid only pre-LN). **Verified** (`circuit.py:554–566`). The repo's validated path-patch
  primitives (`clamp_write`/`capture_writes`, `circuit.py:570–599`) patch **node output tensors** (attn/MLP
  writes), never a Fourier band of the input position embeddings. No primitive subtracts "the carrier's
  downstream residual contribution" at a mid-layer edge, and post-LN means one cannot be built by additive
  subtraction. **The single load-bearing net-new hook is not constructible as specified.**

### F-2 — The only definable operation bundles the direct edge with comb-collapse content routing.
- The one implementable route is: inject into L2MLP's input the layer-2 residual **captured from a globally
  carrier-deleted forward**. That residual already encodes every upstream carrier-reader's altered output.
- **L2H15 is itself a layer-2 period-3 comb head** (`run_carrier_surgery.py` docstring L12, R(k) comb;
  `run_path_decomposition.py:54` `L2,H2=2,15`) whose layer-2 **attention runs before L2MLP in the same layer**
  and writes acceptor-local content into exactly the residual L2MLP reads. Deleting the band collapses comb
  amplitude to **3.4%** (`run_carrier_surgery.py:26–27`), so L2H15's write is materially altered. **Verified.**
- Therefore `ΔΔ_bridge = (direct carrier→L2MLP) + (carrier→comb-collapse→content→L2MLP)`. The reviewer asked
  for ACDC-style patching that distinguishes DIRECT from REDUNDANT routes (Conmy et al. 2023); this design
  isolates neither. The "edge/path restriction that separates B4 from re-measuring C8 globally" (prereg L84–88)
  **does not exist**: the only implementable estimand is the **total upstream carrier effect delivered to
  L2MLP** — i.e. a *localized C8 read*, the very thing §5/M-edge disavow.

### F-3 — M-edge, the memorial the design elevates to "the whole identifiability story," is vacuous w.r.t. the confound.
- M-edge (prereg L166, L174–176) checks only **off-edge bit-identity** (no non-L2MLP read changes, <1e-6). A
  pure forward hook satisfies this **by construction** while injecting a fully confounded on-edge value (the
  collapsed-comb content of F-2). It **cannot fail on the real threat** (direct vs through-comb mediation),
  because both are bundled into the single injected tensor. An on-edge *purity* memorial that could fail —
  hold L2H15/L1/L0 comb outputs at intact while swapping only the input-carrier subspace, require the injected
  delta to lie in the carrier subspace — **cannot be constructed** (post-LN mixing, F-1). So the design certifies
  identifiability with a memorial that passes even when ΔΔ_bridge is 100% backdoor. This is a
  formalize-audit-verify memorial defect (off the path of the real threat), R21-class.

**Net:** the direct/bridge estimand is definitionally entangled with upstream carrier-reader mediation on a
frozen post-LN checkpoint. The controls do not rescue it: **B4.b** (energy-matched E band) separates
"carrier band" from "energy-matched band" but the comb-collapse confound is **itself period-3-specific**, so it
is present for BAND and absent for E — passing B4.b certifies "carrier-specific" while fully consistent with the
effect being 100% comb-collapse content routing. **B4.c** (decoy MLP) is chosen as a *non-B* write, so it has no
partner-conditioned write to gate; ΔΔ_decoy≈0 passes trivially without ruling out that L2MLP's own ΔΔ is
through-attention. Neither separates direct-carrier from through-comb-collapse.

---

## Additional majors (fixable, but they do not lift the fatal)

- **R20 identity-vs-geometry over-read (Endpoint A).** Endpoint A concludes "the positional-addressing
  machinery and the payload-writing MLP **share** a common positional coordinate / the **same** address L3H1
  reads." The intervention never touches L3H1 (layer 3, strictly downstream of L2MLP); it only removes the
  carrier from L2MLP's input. "Same coordinate L3H1 reads" is unverified by the assay — a shared Fourier band
  is not a shared/used address (exactly the R20 trap). Reachable statement is at most "L2MLP's input contains
  the period-3 band," with L3H1 as motivation only. **Verified** (prereg L63–69, L253–257; L3H1=layer 3).

- **No failable on-channel planted-positive (calibration).** The estimand's real channel is
  `edge carrier deletion → L2MLP nonlinear write → hard margin`. PP-c8/M-c8 travels a **different readout**
  (coupling, not margin) at a **different locus** (global layer-0, not the L2MLP edge) — it validates the
  deletion op, not the estimator's sensitivity on the bridge channel. PP-bridge is a **synthetic injection**
  that multiplies the partner-scrub source by the carrier-deleted mask at the edge — it forces the arithmetic
  the estimator computes and **cannot fail unless the estimator is broken**, so it calibrates nothing about
  detecting a real nonlinearly-mediated carrier dependence. Consequence: the MDE of the "locally feasible"
  one-sided positive is uncalibrated; a non-firing result is uninterpretable (insensitive vs no-bridge) — the
  exact P3.4 dead-PP lesson the prereg cites. (Because F-1/F-2 make the on-channel PP inconstructible, this is
  itself further evidence for DEFER, not freeze.)

- **Intent-drift, disclosed but material.** The branch INTENT is "wire the L3H1/carrier → L2MLP bridge," but
  the frozen estimand ΔΔ_bridge = B_I − B_D contains **no L3H1 term** (L3H1 is "named only as the co-reader,"
  B4-1). With a single global period-3 carrier, "L2MLP reads the carrier" and "L3H1 reads the carrier" are
  shared only definitionally; a positive establishes carrier-dependence of L2MLP's write, not any
  L3H1-specific or distinguishable shared-coordinate relationship. The deliverable cannot support the branch
  headline ("bridge"), only "L2MLP's partner write is carrier-gated."

- **Power rationale mislabels a data-location problem as a power floor (statistics).** §7 borrows the P3.4
  "2–4× variance inflation / same wall" number *by analogy, not measurement*, and it **contradicts the
  estimand's own structure**: ΔΔ_bridge is defined as a **paired within-target** second difference (B_I−B_D on
  the same target, differing only by the carrier edge). Paired variance is Var(B_I)+Var(B_D)−2Cov; a targeted
  edge deletion is highly correlated across arms, so pairing typically **reduces** variance (factor
  √(2(1−ρ))<1 for ρ>0.5), not inflates it 2–4×. The C8 anchor (`exon_damage_ci=[0.15551,0.19506]`, HW 0.0198
  at n=384, **verified**) is a **between-gene unpaired raw mean of MLM-ΔL** — a different observable, cohort,
  and pairing — so it does not bound a paired hard-margin B-contrast. The HCA20 whole-B donor CI is already
  **[0.309,0.379] → HW 0.035 < SESOI 0.05** (CLAIMS.md), so a paired second difference off that base is
  plausibly HW<0.05. P3.4's wall was *fundamental* (between-gene, 13–27 genes/frame-class); **B4's is merely
  "the powered cohort layout is remote."** Do not carry the borrowed number; measure the actual gene-clustered
  ΔΔ_bridge SD in a discovery pilot (the MDE gate already scaffolds this). *(This correction does not help the
  bridge: even a well-powered ΔΔ_bridge measures the confounded total-effect of F-2.)*

## Minors (fixable)

- **In-repo cohort is more capable than the draft and the statistics lens assumed — verified.** The statistics
  lens inspected `atlas_context_matched_v7.validation.sequences.jsonl` (3512 rows, **no** gene_id) and
  concluded a GTF gene-label join is net-new before any gene-clustered inference. But
  `atlas_context_matched_v7.validation.packets.jsonl` (**4019 rows, 1486 distinct gene_id, no None**) exists
  in-repo with `gene_id`, `partner_strength`, `target_role∈{donor,acceptor}`, `masked_native_base` — a
  gene-labeled, partner-annotated cohort. **One-way gene-clustered `gboot` is directly runnable locally.**
  Consequence: local feasibility is *higher* than the draft states, which **reinforces** the disposition —
  feasibility is not the blocker, identifiability is. (Caveat: `analysis_split` is a single
  `atlas_internal_validation` value, so B4 must still carve its own gene-disjoint discovery/confirm split.)
- **Dyadic bootstrap mis-imported.** §1/§5/§7 invoke two-way dyadic (sender/receiver) clustering per R22, but
  B4.a–c contain **no interchange**: `partner_destroyed` corrupts the partner motif in the **same** masked
  packet (`run_atlas_model_preflight.py:155`); every estimand is a within-target paired contrast, no gene is
  reused as both an intact and a scrub *source* across targets. **One-way gene gboot is exact and correct
  here.** Remove the dyadic requirement (retain only if a genuine cross-gene interchange arm is later added).
- **B4.b add-one floor under-powered.** `carrier_surgery.json` ships **21 control bands** → add-one p floor
  1/22 = **0.045**; ΔΔ_bridge must strictly exceed all 21, and a single tie gives 2/22=0.091>0.05. Pin N≥39
  (floor 0.025) and construct the extra energy-matched bands in the freeze. **Verified** (21 controls).
- **M-recon is a forced full-restore identity (R21).** `_run_whole_restore` reproducing the native margin in
  the same forward is forced by construction; label it "forced identity (R21), plumbing-only, never evidence,"
  consistent with the M-forced lint row.

## What the draft got right (credit)

- **B4-1 is a genuinely correct, code-verified pre-compute catch.** In a feed-forward post-LN transformer a
  layer-3 head's output cannot be a forward input to a layer-2 MLP, so the literal "L3H1(L3)→L2MLP(L2)" edge
  is architecture-backward and R23-class forced-to-zero — which is exactly why HCA20's same-B attention set is
  empty. **Verified** (`run_path_decomposition.py:53–54`). The draft retires that edge before spending compute.
- The DEFER disposition, the OV/mediation scope cap, the R23/R15/R21 lint (M-forced), and the honest surfacing
  of the Drive-cohort and second-difference concerns are all sound. The design fails only as an *identification
  of a direct carrier→L2MLP bridge*; the intellectual honesty is real.

---

## Feasibility

- **The direct-edge bridge is non-identifiable on a frozen post-LN checkpoint — unfixable by compute.** No
  Colab/CUDA scale, larger n, or new Atlas freeze changes F-1/F-2/F-3. A clean isolation would require freezing
  ALL upstream attention/content at intact while swapping ONLY the positional-carrier subspace in L2MLP's
  input, which presupposes a linear carrier/content separation that does not survive two layers of attention
  mixing under post-LN. That is a **subspace-freeze / training-time** study, out of scope for a frozen-checkpoint
  MPS interp run.
- **The coarse, honest, locally-runnable version is a B3 rider, not B4.** The identifiable question is
  bundled: "is L2MLP's partner-conditioned write carrier-**dependent at all** (through-attention paths
  included), one-sided positive, using the existing global layer-0 `make(BAND)` deletion?" This IS identifiable
  and locally feasible on the in-repo packets cohort (1486 genes, gene-clustered gboot), but it is a
  **total-effect carrier-dependence** claim already adjacent to the C34/C35 carrier account — **it does not
  support the "bridge / shared-address" headline** and belongs folded into B3, not stood up as an Atlas
  headline. It only "matters if the Atlas becomes a headline"; otherwise HCA20's `B` write plus the
  C23/C34/C35 carrier account already cover address × payload.
- **MPS budget note (for completeness):** the local coarse one-sided positive is cheap (~4 arms × a few hundred
  gene-clustered targets × L2MLP restore ≈ 1.6–3.2k forwards, hours on MPS). The *powered equivalence* and any
  HCA20-scale matched confirmation need the Drive a3_v2 cohort (Colab/CUDA) — but per the statistics correction
  this is a **cohort-location** constraint, not a fundamental variance floor.

---

## RECOMMENDATION (to the orchestrator)

**DEFER B4 / fold into B3.** The design red-team did its job: do not freeze or run B4 as a bridge study. The
literal L3H1→L2MLP edge is architecture-backward (B4-1, correctly caught by the draft), and the reframed
"shared-upstream carrier → L2MLP-input" edge is **non-identifiable on a frozen post-LN checkpoint** — the only
implementable carrier deletion is global at the source, and injecting a globally-carrier-deleted layer-2
residual into L2MLP's input bundles the direct removal with comb-collapse content routing through the layer-2
comb head L2H15 that feeds L2MLP in the same block. The memorial the draft leans on (M-edge) is a locality
check that cannot fail on this confound, and the two designed controls (energy-matched E band, decoy MLP) are
orthogonal to it. The blocker is therefore **architectural identifiability, not the power wall or the
Drive-only cohort** — those are real but secondary, and the second-difference "power wall" is in fact a
*paired* contrast off an HCA20 base whose CI already clears SESOI, so it is a cohort-location constraint. In-repo,
`atlas_context_matched_v7.validation.packets.jsonl` already supplies 1486 gene-labeled, partner-annotated
packets, so **feasibility is not the obstacle — identifiability is.** The only locally-runnable, identifiable
variant is a *coarse total-effect carrier-dependence check* on L2MLP's partner write (global layer-0 deletion,
one-sided positive), which does **not** establish a bridge and rides with B3 alongside the existing C34/C35
carrier account. Stand a real cross-estimand bridge study up only as a training-time / subspace-freeze design if
the Head–Context Atlas becomes a headline deliverable. Retain this prereg as design-of-record with the
disposition rewritten to lead with the identifiability fatal.

**Must-fix (documentation-level, before this prereg is retained as design-of-record):** (1) promote the
identifiability non-identifiability to the lead blocker in the disposition; (2) restate the honest reachable
estimand as the **total upstream carrier effect** on L2MLP's write (a localized-C8 read), and withdraw the
"genuinely local narrow bridge version exists" escape hatch; (3) strip "shared with / same address L3H1 reads"
from Endpoint A (R20); (4) reclassify M-edge as a plumbing/locality gate, not the identifiability anchor, and
state explicitly that no memorial in the design can test the direct-vs-mediated carrier confound; (5) correct §7
— drop the borrowed 2–4× number, note the paired second difference and the HCA20 HW 0.035 base, and label the
blocker cohort-location not power; (6) remove the dyadic-bootstrap requirement (one-way gboot exact) and note
the in-repo packets cohort has gene_id; (7) annotate M-recon as forced (R21) and pin B4.b N≥39.

survivor: **NO** — B4 is not executed this session; the identifiable local variant is a coarse B3 rider, not
the B4 bridge.
