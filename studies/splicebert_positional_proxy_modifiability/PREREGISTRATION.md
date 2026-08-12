# Preregistration — SpliceBERT positional proxy modifiability

> **Tier: T2 — provenance.** Frozen design; no model outcome is reported here.

**Study objective, verbatim and frozen:**

> Test whether phase-decoupled continued pretraining weakens SpliceBERT's boundary-register preference, and
> whether identical-sequence relocation shows that the learned period-3 positional carrier causally supports
> that change.

**Status: FROZEN DESIGN 2026-08-11.** The question, geometry, operator, sign convention, estimands,
thresholds, control families, multiplicity rules, gate chain, verdict vocabulary, and outcome language below
are frozen before any model outcome, any pilot, and any continued-pretraining run. The executable package is
**not** frozen by this document: implementation may be exercised on synthetic fixtures and the declared
development role, and nothing may be trained, sealed, or opened until the gate chain in Section 14 passes and
a freeze package binds the final code and inputs. A change after this date is a dated, additive amendment.
This file is never silently edited to agree with an observed result.

---

## 1. What this study inherits, and what it must not

### 1.1 Predecessor

The design conventions and the executable primitives come from
[`../splicebert_positional_boundary_coherence/PREREGISTRATION.md`](../splicebert_positional_boundary_coherence/PREREGISTRATION.md)
§9 (span-wide period-3 register relocation) and from the active modules
[`../../src/gfmcd/splicing/pbc_register_relocation.py`](../../src/gfmcd/splicing/pbc_register_relocation.py),
[`../../src/gfmcd/splicing/pbc_register_runner.py`](../../src/gfmcd/splicing/pbc_register_runner.py), and
[`../../src/gfmcd/splicing/pbc_register_stage.py`](../../src/gfmcd/splicing/pbc_register_stage.py).

That study is **CLOSED and read-only**. Nothing under `studies/splicebert_positional_boundary_coherence/` or
`results/splicebert_positional_boundary_coherence/` is edited, amended, extended, or reopened by this
protocol.

| Inherited | Not inherited |
|---|---|
| Coordinate conventions, window definitions, operator algebra, statistical idioms, tolerances, schema discipline, audit contract, and the live primitives named above | Every numeric reading, every gate value, every development or preflight measurement, every authorization class, and every seal issued under that protocol |

**Blacklist.** All predecessor development, preflight, smoke, and candidate readings — and the `C44`
development preflight, its local two-support surgery, its `Ω`, and its A/V and Q/K signs — are **calibration**.
They are never pooled with, compared to, cited as support for, or used to size, tune, or interpret anything in
this study. This study's own capability numbers are produced fresh on its own development role.

### 1.2 The predecessor's reserved outcome labels are not used here

`P3_DIRECTIONAL_SCOPED_SUPPORT`, `P3_NECESSITY_SUPPORTED`, `P3_REGISTER_RELOCATION_SUPPORTED`,
`P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED`, `P3_REGISTER_PARTIAL_HANDEDNESS_FALSIFIED` / `_AMBIGUOUS`, and
`P3_ROUTE_*` name rungs of the predecessor's frozen hierarchy. They are unearned there and unavailable here.
This study's classifications carry the `PPM_` prefix (Section 15). The one deliberately shared string is
`REGISTER_INSTRUMENT_UNRESOLVED`, whose meaning is fixed in Section 15.3.

### 1.3 Controlled vocabulary

Read [`../../docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md) before writing any prose about this study.
Four distinct objects are in play and an unqualified "phase" is **forbidden** in every outcome sentence:

| Symbol | Object | Kind |
|---|---|---|
| `p` | GENCODE-annotated coding-boundary phase (`0/1/2`) of an authentic exon edge | a property of the annotation |
| `q` | **designed register** of a module parent, `q = shift mod 3` (Section 6) | a property of this study's construction |
| carrier phase | the learned phase of the period-3 component of SpliceBERT's position table | a property of the model |
| `r` | **carrier register**, the imposed intervention coordinate `r ∈ {−1,0,+1}` (Section 7) | an intervention, not an observation |

`δ_design = (q_A − q_D) mod 3` is a designed-register difference. `δ_ann = (p_A − p_D) mod 3` is an
annotated-phase difference. They are different variables measured on different panels and are never merged,
averaged, or substituted for each other.

The maximum interpretation this design can license is a statement about a masked-language model's prior over
joined boundaries and about the modifiability of that prior by a corpus manipulation. It never becomes
"SpliceBERT reads frame", "the spliceosome checks phase", a translation claim, or a claim about U2AF or any
other factor. Attribution ranks; intervention decides. No result at one rung is promoted to the next.

### 1.4 Notation

- `[a, b]` denotes a **confidence interval**, or the frozen `±0.05` equivalence band where that is stated
  explicitly. Spread across lineages, seeds, or bootstrap draws is written "range" or "SD" and never in
  bracket notation.
- `[x, y)` with a half-open bracket denotes a coordinate span, always half-open, always 0-based.
- All logit quantities are **raw masked-MLM logit units**, never nats. "Nats" is reserved for cross-entropy
  damage.

---

## 2. Model, checkpoint firewall, and geometry

### 2.1 Checkpoint

- Base model: official `SpliceBERT.1024nt`; `pytorch_model.bin` sha256
  `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`.
- Architecture: six layers, 16 heads, hidden size 512, post-LayerNorm BERT MLM, **learned absolute position
  table of 1,026 rows × 512 hidden dimensions**.
- The stage-1 510-nt pair and the five GTEx fine-tuned checkpoints are provenance only. No outcome in this
  study may come from them.
- The only other checkpoints that may produce an outcome are the continued-pretraining checkpoints defined in
  Section 16, each of which is derived from the base checkpoint above and records its parent hash.

### 2.2 Cell geometry

Every scored construct is a **cell**: 1,022 nt of nucleotide sequence, tokenized to 1,024 tokens including
`[CLS]` at token 0 and `[SEP]` at token 1,023.

| Object | Nucleotide span | Token span |
|---|---|---|
| donor module (authentic donor parent) | `[0, 394)` | `[1, 395)` |
| donor motif `GT` (masked `G` at nt 392) | `[392, 394)` | `[393, 395)` |
| scaffold / bridge | `[394, 510)` | `[395, 511)` |
| scaffold decoy `AG` | `[435, 437)` | `[436, 438)` |
| acceptor motif `AG` | `[510, 512)` | `[511, 513)` |
| acceptor module (authentic acceptor parent) | `[510, 1022)` | `[511, 1023)` |
| **donor→acceptor span** | `[392, 512)`, length **120** | `[393, 513)`, length **120** |

**The nucleotide→token offset is `+1`, and it is `[CLS]` that creates it.** Every span used by the carrier
operator is stated in both coordinate systems in this protocol, and the executable binds the **token** span.
Section 7.3 states why an error of one here is not detectable downstream.

Motifs, masked position, seam coordinates, sequence length, scaffold content, and tokenized coordinates are
**identical across all nine cells of a block**. Only the authentic module content differs.

### 2.3 Model-native readout

With the donor `G` masked at nucleotide 392 (token 393), the readout is the **linear masked-donor margin**

```
m = logit(G) − mean( logit(A), logit(C), logit(T) )
```

evaluated at the masked donor token.

**This is not the `max` form, and the substitution is not permitted.** The native donor-circuit idiom
`logit(G) − max(logit A, logit C, logit T)` is a non-smooth functional of the three alternatives: its value
changes when the *identity* of the arg-max alternative changes, independently of how much the intervention
moved any logit. Every primary endpoint here is a **difference in differences** — across corruption states,
across `δ` classes, across carrier worlds `r`, and across training arms — and none of those comparisons holds
the arg-max alternative fixed. Under `max`, a contrast would therefore partly measure arg-max switching
between cells rather than the intervention. The linear form is an affine functional of the logit vector, so
every difference of it is exactly the corresponding difference of the underlying logit changes. The `max`
form may be recorded as a descriptive companion; it may never carry an endpoint.

### 2.4 Corruption operator

The corruption is **`AG→AC`** at the acceptor motif `[510, 512)`. Intact and corrupted cells are otherwise
byte-identical, of identical length, with an identical mask.

Per [`../splicebert_partner_selection/CORRECTIONS.md`](../splicebert_partner_selection/CORRECTIONS.md)
**R44**, a two-base substitution constrains only the register it overwrites and can *create* a splice motif in
either flanking register; `AC` is one of exactly two exhaustively motif-clean alternatives and plants nothing
at either site in every cohort measured there.

**`AG→CC` is a different experiment and its numbers do not compare.** No effect size, anchor, threshold,
ratio, or calibration measured under `AG→CC` — including the donor arc's `C1`/`C2` coupling magnitudes — may
be imported into this study, quoted beside one of its endpoints, or used to size it. The two operators are not
interchangeable at the precision this study works at.

Cell coupling, per cell and per carrier world `w`:

```
G_cell(w) = m_intact(w) − m_AG→AC(w)
```

Both raw margins are retained in the artifact, not only their difference (Section 18).

### 2.5 Materiality scale

**SESOI = 0.05 raw masked-MLM logit units**, for every endpoint in this protocol unless a section names a
stricter value. A "bounded null" is an interval **wholly inside** `[−0.05, +0.05]` accompanied by a projected
`MDE80 ≤ 0.05`. A point estimate near zero is not a bounded null and is never reported as one.

---

## 3. Provenance firewall, freshness registry, and seals

### 3.1 Freshness registry

Before any enumeration, build a sorted exclusion registry of version-normalized gene IDs (`ENSG…` truncated
at the first dot) from source artifacts, never from memory. Its union includes:

1. every gene in every cohort underlying `C43` and `C44`, including the 384-block replication and the `C44`
   development preflight;
2. every gene in the stopped `GFS25` V1 and the completed `GFS25` V2 cohorts;
3. every gene named by the exon-edge source registry frozen in the `C43` protocol; and
4. every gene enumerated by any cohort, census, or candidate package of the **closed** positional
   boundary-coherence study.

Archived-only sources are read through read-only object lookup (`git show <rev>:<path>`); no archived runner,
module, test, or cohort is restored to the active tree. The registry artifact records source revision, path,
source sha256, extraction command, normalized count, and the final sorted-list sha256. If a required source
cannot be independently rederived, **construction stops**.

### 3.2 Chromosome seal

Allowed: `chr1`–`chr7`, `chr10`–`chr22`, `chrX`. `chrY` is excluded. **`chr8` and `chr9` remain globally
sealed** and are forbidden to every census, fallback, development, pilot, training-corpus, and confirmation
command. Insufficient capacity is a stop, not authority to spend the seal. Spending it requires, before
enumeration: written owner approval, a dated additive amendment, a newly named split, and a repository-level
record that the seal is spent.

### 3.3 Role disjointness

Five roles, globally disjoint at the level of the version-normalized gene and of the frozen paralogue cluster:

| Role | Purpose | Sealed until |
|---|---|---|
| `development` | operator mechanics, fixtures, base-checkpoint relocation, health, dose | open from the start |
| `pilot` | variance-only training pilot (Section 17) | Section 14 gate chain passes |
| `confirmation_designed` | the sealed identical-sequence designed-register panel | Section 14 passes **and** Section 17 freezes `(L, B)` |
| `confirmation_real` | the sealed fresh authentic-module panel for `Ψ_real` | as above |
| `corpus` | continued-pretraining source windows | as above |

Role assignment is a deterministic hash of the normalized block identity under a frozen salt, using no
sequence score beyond eligibility and no model quantity. A hash-selected subset of `development` is the
plumbing-smoke role; it is not a sixth split.

### 3.4 Input pins

The execution freeze package records and **verifies by comparison** (a recorder that merely prints a value is
not a gate): hg38 FASTA path/size/sha256 and `.fai` sha256; GENCODE v50 basic and comprehensive GTF
paths/sizes/sha256 and coordinate and phase conventions; tokenizer, config, vocabulary, and model file
inventory and sha256 values for the base checkpoint and for every continued checkpoint; the freshness registry
and every input used to derive it; this protocol's sha256; every executable and reusable-module sha256; and
Python, PyTorch, Transformers, NumPy, and SciPy versions plus platform.

---

## 4. Panels

Three panels exist. They are never pooled.

| Panel | Cells | Register variable | Sequences vary across... | Role |
|---|---|---|---|---|
| **Designed panel** | chimeric 3×3 designed-register lattice (Section 6) | `q`, designed | cells within a block; **not** across carrier worlds `r` | carries `Ω3`, `S_r`, `Ψ_design`, `T_clock` |
| **Real panel** | chimeric 3×3 annotated-phase lattice, authentic modules, composition-balanced | `p`, annotated | cells within a block | carries `Ψ_real`, `T_real` |
| **Native health panel** | unjoined authentic source windows | none | — | carries the health channel only (Section 12) |

The real panel is built to the predecessor's inherited construction rules: seven globally unique genes per
block (three donor parents one per `p_D`, three acceptor parents one per `p_A`, one scaffold), the frozen
composition feature family, deterministic role-wise standardized nearest-triplet matching, equal-block
estimand with within-block weights `1/3` on each matched cell and `1/6` on each mismatched cell, maximum
absolute SMD `≤0.10` over the complete three-contrast feature family, and classifier AUC 95% upper bound
`≤0.55` with the planted-label sensitivity and permuted-label specificity gates. Those rules are inherited
**as rules**; none of the predecessor's measured balance values is carried across.

---

## 5. Why this study exists: the two things neither prior panel can do

State this before the design, because every later choice follows from it.

1. **The real panel cannot support a standalone `Ψ` at the 0.05 SESOI.** `Ψ_real` is a within-checkpoint
   matched-minus-mismatched contrast on authentic modules. Powering it to resolve a `0.05` effect on its own
   is a sample-size problem this program has not solved, and a `Ψ` that is statistically resolved but
   sub-SESOI licenses only a rough, assay-scoped statement. A **between-checkpoint difference** on identical
   sequences is a different and easier estimand: every sequence-only confound enters both checkpoints
   identically and cancels exactly in `Ψ_L1 − Ψ_L2`.
2. **The sequence-only designed panel cannot identify the carrier.** See Section 6.4: at a 120-nt span,
   designed-register agreement and joined-window period-3 amplitude are mathematically coupled, so a
   translation-invariant periodicity detector with no positional coordinate reproduces the whole `Ψ_design`
   ranking. Only an intervention that changes the model's positional carrier **while holding the token
   sequence identical** separates the two.

The two co-primary contrasts of Section 19 are exactly these two moves, and neither substitutes for the other.

---

## 6. Phase D — the designed-register lattice (model-free)

Phase D loads no checkpoint. The builder, rotator, balancer, role assigner, sizer, and construction auditor
are forbidden from touching a model.

### 6.1 Block composition

Each block supplies **one authentic donor parent, one authentic acceptor parent, and one scaffold parent** —
three globally unique version-normalized genes, none in the freshness registry, each used in one block and one
role only.

Eligibility (model-free): internal protein-coding exon boundary, unambiguous reference sequence, canonical
`GT`/`AG` as the role requires, adequate flanking support for the fixed geometry, no non-ACGT base in any
measured window, no conflicting local coding feature or strand in the comprehensive-annotation audit, and no
registry gene. Every exclusion is recorded with a reason code.

### 6.2 The frozen rotation

Each module parent yields **three register versions** by a **length-preserving circular rotation** inside its
own module extent. For a module of length `M` placed at cell coordinate `start`, and a shift `s`:

```
cell[start + i] = parent[(i + s) mod M],   i ∈ [0, M)
```

The frozen donor and acceptor motif bases and the seam bases are then re-imposed at their fixed coordinates,
**identically in all three register versions**. The scaffold is not rotated and is byte-identical across all
nine cells of a block.

The three shifts **share one `k`**:

```
s ∈ ( 3k, 3k+1, 3k+2 )        register    q = s mod 3 ∈ (0, 1, 2)
```

Sharing one `k` is what makes the three registers edit-matched: no version is the unrotated parent, and the
three versions differ from each other by at most two nucleotides of displacement.

**The rule that fixes `k`** — applied by the constructor from the pinned geometry and the frozen window list,
not supplied by hand: `k` is the smallest positive integer such that, for both module extents and all three
shifts,

- (i) `3k ≥ L_max`, where `L_max` is the length of the longest frozen measured window; and
- (ii) the circular **wrap junction** — between cell coordinates `start + M − s − 1` and `start + M − s` —
  does not lie inside any frozen measured window, in any register.

Under this protocol's geometry (`M_D = 394` at `start = 0`, `M_A = 512` at `start = 510`) and the inherited
window family (longest window: the 90-nt exonic flanks `[302, 392)` and `[512, 602)`), condition (i) gives
`3k ≥ 90` and condition (ii) gives `s ≥ 92` for the donor module and `s ≤ 420` for the acceptor module. The
unique solution is

```
k = 31        shifts = (93, 94, 95)        q = (0, 1, 2)
```

with donor wrap junctions at cell coordinates `301 / 300 / 299` and acceptor wrap junctions at `929 / 928 /
927` (listed in shift order `93 / 94 / 95`), all outside every measured window. **The constructor recomputes `k` from the rule and fails closed if it
disagrees with 31 or if no `k` exists.** A hand-set `k` is a construction failure.

### 6.3 Lattice, classes, and weights

Crossing three donor versions with three acceptor versions gives all **nine cells** per block. Define

```
δ_design = (q_A − q_D) mod 3 ∈ {0, 1, 2}
```

Exactly **three of the nine cells** fall in each `δ_design` class — forced by the 3×3 crossing. This forced
balance is what makes the cancellation in Section 9.3 exact, so the constructor asserts it and the audit
verifies it, but **it is forced by construction and does not count toward authorization**.

Within-block weights are exact: `1/3` on each of the three cells of the selected class, and `1/6` on each of
the six cells of the two opposing classes. All nine cells of a block travel together through every world,
intervention, and analysis; no cell is ever dropped at runtime.

### 6.4 What the rotation does and does not control

**Forced by the rotation (flagged; not evidence).** A circular rotation is a permutation of the parent's
positions, so the **mononucleotide composition of a module is exactly identical across its three register
versions**, and its di-/trinucleotide composition differs only in the k-mers straddling the moved wrap
junction and the module's linear ends. A composition-balance "gate" on whole-module base content therefore
**cannot fail** and is recorded as a forced identity.

**Not forced (a real, fireable gate).** The frozen measured windows see different parent content in different
registers. Over the frozen composition feature family, computed in the measured windows, the maximum absolute
SMD across the three `δ_design` contrasts (`0` vs `1`, `0` vs `2`, `1` vs `2`) must be `≤ 0.10` separately in
every role. A pair-feature classifier predicting `1[q_D = q_A]` under nested five-fold block-grouped
cross-validation must have ROC AUC 95% upper bound `≤ 0.55`, with a planted-label arm reaching lower bound
`≥ 0.70` (sensitivity) and 99 balanced pseudo-label assignments giving family FPR `≤ 0.05` (specificity).

**Not controlled at all, and the reason `Ψ_design` is only a diagnostic.** The span is **120 nt, a multiple of
3**. When `δ_design = 0` the two modules' intrinsic period-3 sequence structure is in the same register on both
sides of the join, so period-3 sequence energy across the joined window adds **constructively**; when
`δ_design ≠ 0` it adds **destructively**. Designed-register agreement and joined-window period-3 amplitude are
therefore two names for one quantity in this geometry. No balance objective can separate them, because they
are not two correlated features — they are algebraically the same feature.

### 6.5 Construction leakage audit

Model-free, independent, and required before any checkpoint is loaded: rederive every parent window, rotation,
shift, register label, motif, seam, wrap junction, coordinate, sequence sha256, and role assignment from the
pinned inputs; verify global gene and paralogue-cluster disjointness across all five roles; verify the
freshness registry exclusion; verify `chr8`/`chr9` absence; verify that no confirmation or pilot sequence
shares an exact 32-nt substring with any training-corpus window in either direction; verify the exact per-class
cell counts; and verify `k` against the rule. Failure is `PPM_CONSTRUCTION_UNRESOLVED` and no checkpoint is
loaded.

---

## 7. The carrier intervention

### 7.1 What changes and what does not

Across the three carrier worlds `r ∈ {−1, 0, +1}` the **token sequences are identical**. The same nine cells
of the same block are scored three times. Nothing about the input changes: not a base, not a token, not the
mask, not the corruption, not the batch composition.

What changes is the model's **learned period-3 positional carrier**, and only inside the donor→acceptor span,
by a span-limited analytic phase ramp applied through a **forward hook on the position embedding**. Model
weights are never modified; the hook adds a frozen per-batch-element delta to the position-embedding output and
is removed on exit. `r = 0` is an **exact no-op**: the delta tensor is identically zero.

### 7.2 The operator

The period-3 band is **rFFT bins 339..345** of the 1,026-row position table; the centre bin is
`1026 / 3 = 342`, exactly period three. For token row `j`, the positive-frequency analytic band component is

```
Z3(j) = (2/N) · Σ_{κ=339}^{345} P̂_κ · exp( i·2πκj/N ),      N = 1026
```

evaluated **continuously from the frozen Fourier coefficients** in float64 — never by interpolating or rolling
position-table rows — and cast to the embedding dtype only after evaluation.

The ramp over the token span `[393, 513)` is

```
φ_r(j) = −(2πr/3) · (j − 393) / 119
```

and the edited table is

```
P_r(j) = P(j) − Re[Z3(j)] + Re[ Z3(j) · exp(i·φ_r(j)) ]     for j ∈ [393, 513)
P_r(j) = P(j)                                                otherwise, exactly
```

So the ramp is exactly `0` at token 393 (the masked donor `G`, nucleotide 392) and exactly `−2πr/3` at token
512 (the acceptor `G`, nucleotide 511, the second base of the acceptor `AG`). Every non-period-3 component of
every row, the row mean, and every row outside the span are retained algebraically. The clipped edit is not
claimed to be globally band-limited; the expected global spectral leakage of a windowed phase ramp is
**reported and matched by the controls**, not mislabelled a failure.

Executable binding: `carrier_edit_delta(..., phase_mode="relative", direction="right")` applied through
`embedding_delta_context` on `bert.embeddings.position_embeddings`.

### 7.3 The off-by-one, stated explicitly

The span in **nucleotide** space is `(392, 512)`. The span in **token** space is `(393, 513)` — **one greater
at both endpoints, because `[CLS]` occupies token 0.** The executable binds the token span.

An error of one row at either endpoint displaces the ramp by exactly one position-table row. **At period three,
one row is exactly `2π/3` of carrier phase, which is exactly one register step.** An off-by-one therefore does
not produce noise, a partial effect, or an inconsistency: it relabels every carrier world by one step and
yields a fully self-consistent lattice with the **wrong handedness**. No downstream statistic can detect it —
`Ω3`, every `S_r`, every control contrast, and every interval remain internally coherent. **Only the pure-tone
audit of Section 14 gate 2 catches it**, because that audit alone measures the operator's displacement against
a synthetic carrier of known phase at known rows. This is the single highest-consequence mechanical failure in
the protocol and it is gated accordingly.

### 7.4 Mechanical invariants

For every world and every control, before any development interpretation:

1. `r = 0` produces an identically zero delta (**forced by the operator definition; flagged, not evidence**),
   **and** the hooked `r = 0` forward reproduces the unhooked native forward to `≤ 1e-8` in every logit
   (fireable: a hook can corrupt a batch even when its delta is zero);
2. rows outside the bound span are byte-identical, and the pointwise non-period-3 residual `P − Re[Z3]` is
   unchanged on the span within `1e-10` relative;
3. the measured analytic phase at token 393 is `0` and at token 512 is `−2πr/3`, each within `1e-6` rad, and
   the intermediate rows are monotone in signed phase;
4. edit-application order, batch order, batch size, and corruption state change no result above `1e-8`, and
   repeated deterministic forwards differ by at most `1e-8`;
5. token IDs, nucleotide text, masks, motifs, corruption pairing, cell labels, and coordinates are identical
   across all carrier worlds — asserted by hashing the tokenized batch in each world and requiring equality;
6. every named control edit matches the target edit's changed-table RMS within `1e-10` relative; and
7. all logits, losses, and derived statistics are finite.

---

## 8. Frozen handedness

### 8.1 The sign convention

```
delta = (−r) mod 3
```

Full truth table, frozen before any data:

| `r` | `−r` | `delta = (−r) mod 3` | predicted preferred class | the two mismatched classes | opposite-handed prediction `(+r) mod 3` |
|---:|---:|---:|---|---|---|
| `−1` | `+1` | `1` | `δ_design = 1` | `{0, 2}` | `2` |
| `0` | `0` | `0` | `δ_design = 0` | `{1, 2}` | `0` |
| `+1` | `−1` | `2` | `δ_design = 2` (i.e. `−1`) | `{0, 1}` | `1` |

This is the convention implemented by `preferred_delta(r) = (−r) % 3`, and it follows from the
positive-frequency analytic convention of Section 7.2: the negative ramp evaluates a pure centre-bin carrier
as `C3(j − r)` at the acceptor endpoint, giving source displacement `−r`, carrier-register label `r`, and
effective-state equality `delta + r = 0`.

### 8.2 The `r = 0` world cannot decide handedness

Both conventions predict `δ_design = 0` at `r = 0`. The agreement there is **forced** and contributes exactly
zero to every handedness contrast (Section 9.4 proves the cancellation). Handedness is therefore decided by
`r = −1` and `r = +1` only, and the `r = 0` term is excluded from the handedness family rather than counted in
it.

### 8.3 Opposite-handedness is a falsification path, not a fallback

An opposite-handed result that clears all gates is a **prespecified falsification** of the frozen orientation
and is classified `PPM_HANDEDNESS_FALSIFIED` (Section 15). It is not renamed positive, not described as "the
effect with the other sign", and not merged with the expected orientation into a two-sided claim.

**The sign may never be relabelled after outcomes.** Not by re-deriving the convention, not by re-reading the
Fourier convention, not by "the ridge is clearly at the other offset". The convention above, the pure-tone
audit that verifies it, and the executable that implements it are hash-bound together in the freeze package
before any panel opens.

---

## 9. Primary endpoints

### 9.1 Definitions

Let `G(delta, r)` be the mean per-cell coupling `m_intact − m_AG→AC` over the cells of class `δ_design = delta`
in carrier world `r`, formed blockwise with the exact weights of Section 6.3 and then equal-block averaged.

```
S_r    = G((−r) mod 3, r) − mean over delta ≠ (−r) mod 3 of G(delta, r)

Omega3 = mean over r ∈ {−1, 0, +1} of S_r
```

### 9.2 `Psi_design` is `S_0`, and it is DIAGNOSTIC ONLY

```
Psi_design = G(0, 0) − [ G(1, 0) + G(2, 0) ] / 2  ≡  S_0
```

`Psi_design` is not an independent quantity: it is exactly the `r = 0` member of the register family, measured
in the unhooked native world.

**It is retired as an identifier of the carrier.** It remains a valid descriptive contrast on the designed
panel and is reported as one; it may never be cited as evidence that the model used its positional carrier,
and it may never carry a claim on its own. The reason is Section 6.4: at a 120-nt span, 120 is a multiple of 3,
so designed-register agreement and joined-window period-3 **amplitude** are mathematically coupled — matched
registers add constructively and mismatched registers add destructively. A translation-invariant periodicity
detector with no positional coordinate whatsoever reproduces the entire `Psi_design` ranking. Only the
`r`-intervention, which holds the token sequence identical and changes only the model's learned positional
carrier, separates the two.

Consequently, no gate, classification, or licensed sentence in this protocol rests on `Psi_design` alone, and
`S_0` alone never satisfies a specificity requirement.

### 9.3 Why `Omega3` identifies where `Psi_design` cannot

Fix `r` and define, for each class `delta0`,

```
T_r(delta0) = G(delta0, r) − (1/2) · Σ_{delta ≠ delta0} G(delta, r)
```

Then `S_r = T_r((−r) mod 3)`, and for any fixed `r` the three orientation contrasts satisfy the exact identity

```
T_r(0) + T_r(1) + T_r(2) = 0
```

**Therefore, under any readout that does not depend on `r` — that is, under the sequence-only rival of
Section 6.4, or under any model whose behaviour on these cells is unaffected by the positional-carrier edit —**

```
Omega3 = (1/3) · [ T(1) + T(0) + T(2) ] = 0     exactly
```

because the three `S_r` select the three *different* classes and their sum telescopes to zero. `Omega3` is
non-zero only if `G(delta, r)` genuinely depends on `r`. This exactness relies on the forced per-class cell
balance of Section 6.3 and on the equal weights there; both are asserted by the constructor and verified by
the audit.

Two consequences are recorded rather than left implicit:

- the sum-to-zero identity means a positive `S_r` **mechanically** forces the mean of the other two
  orientations at that `r` to be negative, so "expected orientation beats the mean of the others" is a forced
  anti-correlation and **is not by itself evidence of handedness**. Handedness needs the named contrast `H`
  of Section 9.4 and the named controls of Section 10; and
- `Omega3 > 0` under a failed or mis-specified operator is meaningless, which is why the mechanical and
  pure-tone gates precede every reading of it.

### 9.4 Handedness contrast

```
Sopp_r  = G((+r) mod 3, r) − mean over delta ≠ (+r) mod 3 of G(delta, r)
Omega_opp = mean over r of Sopp_r
H       = Omega3 − Omega_opp
```

Because `Sopp_0 = S_0` identically, the `r = 0` terms cancel exactly and

```
H = (1/3) · [ (S_{−1} − Sopp_{−1}) + (S_{+1} − Sopp_{+1}) ]
```

which is Section 8.2 in algebra: `H` is carried entirely by `r = ±1`.

### 9.5 Between-checkpoint contrasts

For the two continued-pretraining arms of Section 16, computed **per lineage** as a paired within-lineage
difference and then averaged over lineages:

```
T_real  = Psi_L1  − Psi_L2      on the sealed fresh authentic-module real panel
T_clock = Omega3_L1 − Omega3_L2 on the sealed identical-sequence designed panel
```

`Psi` on the real panel is the inherited annotated-phase matched-minus-mismatched contrast
`M0 − (M+ + M−)/2` under the same linear readout and the same `AG→AC` corruption.

---

## 10. Controls

Every control edit is **RMS-matched to the target dose** — rescaled so that its changed-table RMS equals the
target edit's changed-table RMS within `1e-10` relative — and is executed for every non-zero register. At
`r = 0` every control is an exact no-op, like the target.

| Control | Definition | What it rules out |
|---|---|---|
| **Common-mode phase** | **one uniform angle across the span**, not a ramp: `φ(j) = −2πr/3` for every `j ∈ [393, 513)` | that any period-3 phase perturbation of matched size moves the lattice; isolates the *relational* ramp from a bulk rotation |
| **Decoy span** | the identical ramp on a length-matched **disjoint** span, nucleotides `[152, 272)` / tokens `[153, 273)` — 120 rows, present in every construct, disjoint from `[392, 512)` | that the effect is a generic consequence of editing 120 position rows anywhere |
| **Opposite-handed** | the ramp with the opposite sign convention, predicting `delta = (+r) mod 3` | the frozen orientation; supplies `H` (Section 9.4) |
| **Signed hidden-axis permutations** | **31** frozen signed permutations of the period-3 component's hidden axes, seeds `2026081201`–`2026081231`, each RMS-matched; band power and the spatial Gram structure preserved | that any matched-size, matched-spectrum edit in the same band moves the lattice |

**Requirements.** Learned `Omega3` must exceed **every** one of the 31 hidden-axis controls, with exact
add-one one-sided `p = (1 + #controls at least as extreme)/32 ≤ 1/32` and a paired
learned-minus-mean-control 95% lower bound `> 0`. No Gaussian tail is ever substituted for the finite control
family, and the `p`-value floor `1/32` is reported as granularity, not as strength.

The **carrier-specificity family** is `{ H, Omega3 − Omega3_common-mode, Omega3 − Omega3_decoy }`, evaluated at
block level with max-studentized simultaneous 95% lower bounds; all three must exceed zero. Additionally,
**neither the common-mode nor the decoy control may itself satisfy the target's raw positive rule** across its
own three `S_r`. This is a failure-to-reproduce requirement plus a positive paired separation; it is not an
equivalence test and is never reported as one.

---

## 11. Multiplicity, inference, and resolution

### 11.1 Units and resampling

- The independent unit for a **within-checkpoint** endpoint (`Omega3`, `S_r`, `H`, `Psi_design`, `Psi_real`) is
  the **block**. Resample 20,000 chromosome-by-strand whole-block bootstrap draws, NumPy `PCG64`, seed
  `2026081102`.
- The independent unit for a **between-checkpoint** endpoint (`T_real`, `T_clock`) is the **lineage** — one
  paired `(L1, L2)` continuation from the base checkpoint under one seed. Blocks are **nested inside**
  lineages. Resample two-level: lineages with replacement at the top level, then blocks with replacement
  within each drawn lineage, 20,000 draws, seed `2026081103`. Studentize with the two-level standard error.
- No runtime exclusion, winsorization, imputation, or per-world missingness. A failed forward makes the
  one-shot run anomalous and blocks interpretation.

### 11.2 Families

| Family | Members | Rule |
|---|---|---|
| **Co-primary conjunction** | `{T_real, T_clock}` | **Intersection–union test.** Both one-sided 95% lower bounds must exceed zero. Because the end-to-end claim requires **both** rejections, the conjunction is valid at level `0.05` with **no** multiplicity adjustment; taking one would be conservative and is not taken. |
| **Standalone lower rung** | `T_real` alone, or `T_clock` alone | Holm across the two-member family: the smaller `p` is compared at `0.025`. A standalone rung never uses the unadjusted conjunction level. |
| **Register family** | `{Omega3, S_{−1}, S_0, S_{+1}}` | Max-studentized simultaneous 95% one-sided lower bounds. |
| **Carrier-specificity family** | `{H, Omega3 − common-mode, Omega3 − decoy}` | Max-studentized simultaneous 95% one-sided lower bounds; all `> 0`. |
| **Structured-null family** | 31 signed hidden-axis controls | Exact add-one one-sided `p ≤ 1/32` **and** paired lower bound `> 0`. |
| **Health** | Section 12 | **Fail-closed pass/fail. Never a multiplicity member.** A failed health gate stops the run; it is not "corrected for". |

The register and carrier-specificity families are corrected separately and neither taxes the co-primary
conjunction.

### 11.3 Projected MDE rule

```
MDE80(n) = ( z_0.975 + z_0.80 ) · SD_hat / sqrt(n) = 2.801585 · SD_hat / sqrt(n)
```

with `SD_hat` computed from **stratum-centered** units (each chromosome-by-strand stratum centred before the
pooled SD, matching `projected_mde80` in the live module; a stratum with fewer than two units is an error, not
a skip).

- For a within-checkpoint endpoint, `n` is the block count and `SD_hat` the stratum-centered block SD.
- For a between-checkpoint endpoint, the projection is **two-level**:

  ```
  MDE80(L, B) = 2.801585 · sqrt( σ²_lineage / L  +  σ²_block / (L · B) )
  ```

  with `L` paired lineages and `B` blocks per lineage. **A block-level projection is forbidden as a
  substitute**, and a block-level interval may never be quoted for `T_real` or `T_clock`.
- The Monte-Carlo standard error of every projected MDE must be `≤ 0.0002`.

**The floor that matters.** `MDE80(L, B) ≥ 2.801585 · σ_lineage / sqrt(L)` regardless of `B`. Adding blocks,
adding cells, or adding synthetic register variants cannot reduce the lineage term. **Blocks and synthetic
variants do not substitute for training replicates.** Any design that meets `MDE80 ≤ 0.05` only by inflating
`B` at fixed `L` fails gate 7.

---

## 12. Health

### 12.1 Masked probe health only

The general-competence channel is a **masked-base** probe. **The probe positions must be masked.**

An earlier unmasked probe in this program read the arg-max at random *unmasked* positions and compared it to
the base sitting right there in the input — so it scored the model's ability to copy its own input. It
returned about `0.99` for everything, including **`0.998` for a model with 94 of 96 heads ablated**, which was
*above* its baseline. **It could not have failed**, so it never licensed anything it was cited for. That is
**R8**. Masking is what makes this a measurement, and this protocol admits no unmasked competence probe in any
role, including as an informal sanity print.

### 12.2 The health channel is never the outcome channel

Cite the **R13** error: a gate once used `min(exon, intron)` headroom while *exon headroom was the outcome*, so
the health check and the measured quantity shared a channel and the gate could not separate "the intervention
killed the effect" from "the intervention killed the model" — nor could it fail independently of the result it
was guarding.

Accordingly:

- masked probe positions are drawn **outside** the donor→acceptor span `[392, 512)`, outside the decoy span
  `[152, 272)`, and away from the masked donor token; they never include the readout position;
- the donor-coupling health gate is measured on the **native health panel** — unjoined authentic source
  windows — never on designed or real cells; and
- no health gate reads `Psi_design`, `Psi_real`, `S_r`, `Omega3`, or any contrast derived from them.

### 12.3 The gates

Every checkpoint, and every position-table world that may be interpreted, must pass:

1. **masked-base accuracy retention `≥ 0.80`** relative to the unedited base-checkpoint table on the identical
   masked probe positions, with the retention 95% lower bound `> 0.80`;
2. **protected true-base log-probability change 95% lower bound `> −0.10` nat** on the same masked probes;
3. **native-source `AG→AC` donor coupling 95% lower bound `> +0.05`** on the native health panel;
4. **fixed-bridge `AG→AC` effect** — corrupting the scaffold decoy `AG` at `[435, 437)` — interval wholly
   inside `±0.05`;
5. no block, cell, or gene fails in only one world; a numerical failure excludes nothing and stops the run;
   and
6. for the training arms only: the **paired L1−L2 general masked-base accuracy difference** 95% interval
   wholly inside `±0.02`. This is the arms' competence-matching gate. If it fails, the arms differ in general
   competence as well as in corpus register structure, the co-primary contrasts are confounded by that
   difference, and the result is `PPM_ARMS_NOT_COMPETENCE_MATCHED`.

Gate 6's `±0.02` margin is a **chosen** equivalence band on a probability-scale channel; it cannot be derived
from the `0.05` logit-unit SESOI and is recorded here as a design choice, fixed before any arm is trained.

Health gates may be used for dose selection. They may not be used to select an outcome, a checkpoint step, or
a cohort.

---

## 13. Fixtures

Six fixtures must traverse the **real** analysis and decision functions — not a reimplementation of the
decision rule inside the test — before any development interpretation:

| Fixture | Construction | Required behaviour |
|---|---|---|
| planted positive | synthetic couplings with a `+0.075` effect planted on the expected orientation | classified positive; corrected power `≥ 0.80` |
| opposite-handed | the same effect planted on `(+r) mod 3` | classified `PPM_HANDEDNESS_FALSIFIED`, **never** positive |
| no-ramp | `r` worlds populated with identical `r`-invariant couplings | `Omega3` estimate exactly `0` to machine precision (this is the Section 9.3 identity; **forced, flagged**) and the decision is non-positive (**fireable**) |
| wrong-support | the ramp applied to a span offset by one token | must **not** reproduce the target classification; the pure-tone audit must flag the support |
| common-mode | a uniform-angle edit at matched RMS | must not satisfy the target rule |
| null | sign-symmetric synthetic replicates, 1,000 draws | empirical familywise false-positive rate `≤ 0.06` |

---

## 14. The gate chain

Eight gates, **in this order**. Each row is **fail-closed**: a failure stops every dependent stage, and the
dependent panel, corpus, or lineage stays sealed. Nothing downstream of a failed gate may be run "for
information".

| # | Gate | Passes when | Failure |
|---|---|---|---|
| 1 | **Model-free construction and leakage audit** | Section 6.5 in full: every sequence, rotation, register label, motif, seam, junction, coordinate, hash, role disjointness, registry exclusion, chromosome seal, corpus non-overlap, per-class counts, and `k` rederived from pinned inputs by an independent auditor that loads no checkpoint | `PPM_CONSTRUCTION_UNRESOLVED` |
| 2 | **Operator mechanics and pure-tone handedness sign** | Section 7.4 items 1–7, **plus** an independently implemented symbolic/index audit and a **pure-tone audit**: a synthetic table whose only content is a unit-amplitude centre-bin (`κ = 342`) tone of known phase per hidden axis is passed through the operator, the per-row analytic phase is measured, and it must equal `φ_r(j)` within `1e-6` rad at **every** row, with `0` at token 393 and `−2πr/3` at token 512; the deliberately off-by-one support must be **rejected** by this audit | `REGISTER_INSTRUMENT_UNRESOLVED` |
| 3 | **Analysis fixtures** | all six fixtures of Section 13 pass through the real analysis and decision functions; familywise FPR `≤ 0.06`; corrected power `≥ 0.80` at a planted `0.075` | `REGISTER_INSTRUMENT_UNRESOLVED` |
| 4 | **Development-only base-checkpoint relocation** | the complete target-and-control lattice executes on the **base checkpoint** on the **development** role only, with all mechanical and health invariants holding in every world | `REGISTER_INSTRUMENT_UNRESOLVED` |
| 5 | **Development register capability** | on that development reading: **all three `S_r` point estimates oriented as predicted** by the Section 8.1 truth table; **aggregate `Omega3` resolved** with 95% lower bound `> 0`; and **expected handedness superior to controls** — learned `Omega3` exceeds all 31 hidden-axis controls (`p ≤ 1/32`, paired lower bound `> 0`) and all three carrier-specificity simultaneous lower bounds `> 0`, with neither named control reproducing the target's raw positive rule | `REGISTER_INSTRUMENT_UNRESOLVED` |
| 6 | **Masked MLM and donor-coupling health** | Section 12.3 items 1–5 pass for the base checkpoint and for **every** interpretable position-table world including all controls | `REGISTER_INSTRUMENT_UNRESOLVED` |
| 7 | **Projected target MDE** | projected `MDE80 ≤ 0.05`, with Monte-Carlo SE `≤ 0.0002`, for **`T_real`, `T_clock`, `Omega3`, each `S_r`, `H`, and each carrier-specificity contrast**, under the Section 11.3 rule and — for the two between-checkpoint endpoints — the two-level projection at the frozen `(L, B)` | `REGISTER_INSTRUMENT_UNRESOLVED` |
| 8 | **Independent audit and authorization seal** | an auditor that does **not** import the verdict function recomputes gates 1–7 from primitive rows and emits `PASS`; a separate freezer then writes the seal binding protocol, code, checkpoint, construction package, cohort, seeds, thresholds, commands, and immutable output paths, on a clean tracked revision | no seal; nothing opens |

**Gate 4 and gate 5 are development capability readings. They are not results.** They are never reported as
findings, never quoted in prose, never compared to a predecessor number, and never entered in a claim ledger.

**A producer never authorizes its own dependent target.** The runner that measures a gate's inputs may not
emit the authorization consumed by the stage that depends on it. The chain is three-party throughout:
producer → independent auditor (`PASS`) → separate freezer (seal). The target runner recomputes every binding,
refuses a dirty tree, refuses a self-issued audit chain, refuses an unauthorized world, and exits **before
checkpoint loading** on any mismatch.

Gate 4 through gate 8 may be entered only after gates 1–3; gates 1–3 require no checkpoint at all except the
base checkpoint for gate 2's table.

---

## 15. Verdict vocabulary

### 15.1 Classifications

All classifications require the full gate chain, the construction audit, the health gates, the code and input
hashes, one-shot chronology, and an independent result audit.

| Conditions | Classification | Licensed statement |
|---|---|---|
| Both co-primary lower bounds `> 0` under the intersection–union rule; carrier-specificity family all `> 0`; structured-null requirement passes; all health gates pass | `PPM_MODIFIABILITY_SUPPORTED` | Phase-decoupled continued pretraining weakens this checkpoint family's boundary-register preference, and the identical-sequence carrier intervention shows that the learned period-3 positional carrier causally supports that change, in this assay and this corpus manipulation. |
| `T_real` passes at the Holm-corrected standalone level; `T_clock` does not | `PPM_REAL_PANEL_SHIFT_ONLY` | Phase-decoupled continued pretraining shifts the annotated-boundary preference on authentic modules. **No carrier attribution.** Behaviour rung only. |
| `T_clock` passes at the Holm-corrected standalone level; `T_real` does not | `PPM_CLOCK_SHIFT_ONLY` | The corpus manipulation changes the model's response to an identical-sequence carrier intervention on the designed panel. **Nothing is licensed about authentic modules or about the annotated-phase preference.** |
| Both co-primary intervals wholly inside `±0.05` with projected and observed `MDE80 ≤ 0.05` for both | `PPM_BOUNDED_NULL` | This corpus manipulation, at this token budget, does not materially move either endpoint at the `0.05` scale. It does not show that the carrier is unmodifiable. |
| The opposite-handed family passes while the expected orientation does not, with all gates green | `PPM_HANDEDNESS_FALSIFIED` | The frozen orientation is falsified. It is not renamed positive and is not merged into a two-sided claim. |
| Health gate 12.3(6) fails | `PPM_ARMS_NOT_COMPETENCE_MATCHED` | The arms are not competence-matched; the co-primary contrasts are confounded by a general-competence difference and are not interpreted. |
| Any preflight, mechanical, fixture, capability, health, resolution, seal, or audit gate fails | `REGISTER_INSTRUMENT_UNRESOLVED` | See 15.3. |
| Any other pattern | `PPM_INCONCLUSIVE` | Report estimates and boundaries; stop at the strongest rung actually passed. |

### 15.2 Rung independence

`T_real` and `T_clock` sit at different rungs and neither upgrades the other. A `PPM_REAL_PANEL_SHIFT_ONLY`
result is a behaviour-rung statement with **no** mechanistic attribution. A `PPM_CLOCK_SHIFT_ONLY` result is
carrier-specific but confined to the designed panel, where `Psi_design` is a diagnostic (Section 9.2) and
external validity to authentic modules is exactly what `T_real` was there to supply. Only the conjunction
licenses the end-to-end sentence.

### 15.3 `REGISTER_INSTRUMENT_UNRESOLVED`

**This must never be reported as "the ridge does not move."**

**A failed instrument licenses no conclusion of any sign** — not positive, not null, not bounded, not
suppressive, not inverted, not "trending". It is a statement that the declared question could not be asked,
and the only licensed sentence names the gate that failed. It is not evidence about the carrier, about the
corpus manipulation, about the predecessor study, or about any future study. A sealed panel that was never
opened remains sealed and unspent.

---

## 16. The training protocol — frozen, NOT authorized to run

**Nothing in this section may be executed until the Section 14 gate chain passes in full and the owner
authorizes.** It is frozen here so that the training artifact cannot be designed after a preflight outcome is
visible.

### 16.1 Arms

| Arm | Definition | Role |
|---|---|---|
| **L1** | **edit-matched, phase-preserving** continuation: every eligible corpus window receives a length-preserving circular rotation of its downstream coding module by a per-window offset `o` with `o ≡ 0 (mod 3)` | the paired comparator |
| **L2** | **phase-decoupled** continuation: the identical edit with `o ≢ 0 (mod 3)` | the treatment |
| **L3** | *optional* **period-3-destroyed calibration** arm: exonic segments are shuffled blind to codon position at a matched edit count | calibration only |

The per-window offset magnitude `abs(o)` is drawn from a **frozen deterministic hash of the window identity**
whose distribution is **identical across L1 and L2**; only the residue `o mod 3` differs. Boundary motifs are
re-imposed identically in both arms. **L1 and L2 therefore differ in exactly one property of the corpus: the
period-3 register relation across annotated coding boundaries.** They do not differ in edit count, edit size
distribution, window content, or motif content.

This is a **corpus-level manipulation**, not a claim about natural training data. It shows what a model trained
with a decoupled register relation does; it does not establish what the original pretraining corpus caused.

**L3 is a positive control on the training protocol** — it asks whether continued pretraining under this
recipe can move these endpoints at all. It is not competence-matched to L1, its health cost is measured and
reported rather than assumed, it never enters a co-primary contrast, and it can never rescue a failed L1/L2
comparison.

### 16.2 Identity requirements across the paired arms

Identical across L1 and L2, within a lineage: initial checkpoint; source windows; **window order**; masking
randomness (the same RNG stream and the same masked index set at every step); MLM objective and masking rate,
copied from the pinned base-pretraining recipe; optimizer, precision, and determinism settings; **optimizer
schedule**; and **token budget**.

Verification is mechanical, not attested: the realized training stream of each arm is hashed — window IDs,
order, per-step masked index sets, step count, and per-step token count — and the two arms' stream hashes must
agree in every field except the recorded offset residues. A mismatch is a stop.

### 16.3 Corpus exclusions

Excluded from **every** training corpus, in every arm and every lineage:

- every target and pilot parent, scaffold, and assembled cell, by version-normalized gene ID;
- every gene in the same frozen paralogue cluster or orthogroup as any of those;
- every alternative transcript and annotated variant of any of those genes; and
- every **near-identical window**: any corpus window sharing an exact **32-nt** substring with any pilot or
  confirmation parent, scaffold, or assembled cell, or reaching `≥ 90%` identity over `≥ 100` nt under the
  frozen aligner.

The `32`-nt, `90%`, and `100`-nt near-identity constants are **chosen** exclusion thresholds, fixed before any
corpus is built and recorded here as design choices rather than derived quantities. The bidirectional check is
part of the gate-1 leakage audit.

### 16.4 Schedule and checkpoint selection

Continuation hyperparameters are copied verbatim from the pinned base-pretraining recipe, with the peak
learning rate set to the recipe's terminal-schedule value — continuation resumes the recipe's own schedule
rather than re-warming. The **token budget** is frozen in the execution package, before any confirmation
lineage is trained, at the value the pilot's compute envelope permits for `L ≥ 6` paired lineages; it is a
function of available compute only, never of any endpoint, and it is identical across paired arms.

**No plateau-dependent schedules.** A "train until the loss plateaus" rule makes the stopping point a function
of the data the model saw. L1 and L2 see differently structured corpora, so they would plateau at different
steps and receive different budgets, and the contrast would confound treatment with schedule length. The
budget is a constant fixed in advance.

**No target-based checkpoint selection.** Choosing the step whose `Psi`, `Omega3`, `S_r`, or any related
quantity is largest is selection on the outcome; it inflates the contrast without bound and makes the reported
interval meaningless. **Exactly one checkpoint per arm per lineage is read: the final step of the frozen
budget.** Intermediate checkpoints may be retained for provenance; none may be scored on a confirmation panel.

Both prohibitions have the same root: each would let the outcome influence which artifact exists.

---

## 17. The variance-only pilot

### 17.1 Purpose and blinding

The pilot runs on the **`pilot` role — separate development families**, whose parents, scaffolds, and genes are
disjoint from every confirmation panel and from every training corpus. Its sole purpose is to estimate

- the **same-block L1/L2 covariance**, and
- the **between-seed (between-lineage) variance**

**without reading treatment means or signs.**

Blinding is mechanical. Arm labels in the pilot artifact are replaced by a per-lineage random relabelling
`{A, B}` drawn from a sealed permutation; the key is written to a separate sealed file that no analysis command
may read before the confirmation seal exists. The pilot analysis schema has **no field capable of holding an
arm mean, a per-arm endpoint, a signed contrast, or a verdict**, and the pilot auditor verifies both the schema
and the absence of any such field.

Same-block covariance and the between-lineage variance of the paired difference are invariant under this
relabelling; the mean of the paired difference is not, and is destroyed by it. The relabelling inflates the
estimated between-lineage variance by exactly the squared treatment mean, so **the resulting sample size is
conservative** — the blinding can only increase `L` and `B`, never decrease them. This is stated so that the
conservatism is read as intended, not as a defect.

### 17.2 Pilot size

The pilot runs **at least six paired lineages**. Fewer is not permitted: a between-lineage SD estimated on `n`
pairs has relative standard error about `1/sqrt(2(n−1))`, which at `n = 5` exceeds `0.35` — the projection in
Section 11.3 would then be less reliable than the quantity it is sizing.

### 17.3 The freezing rule

Freeze the confirmation **lineage count `L`** and **block count `B`** by this prespecified lexicographic rule,
applied to the blinded variance components:

1. take the smallest integer `L ≥ 6`, then the smallest integer `B`, such that the two-level projection of
   Section 11.3 gives `MDE80 ≤ 0.05` for **`T_real`, `T_clock`, `Omega3`, each `S_r`, `H`, and each
   carrier-specificity contrast**;
2. round `B` **up to the next multiple of three**, so that the three designed registers are exactly balanced
   within every chromosome-by-strand stratum (design-forced, not a guard factor); and
3. recompute the projection at the frozen `(L, B)` and require `MDE80 ≤ 0.05` to hold with Monte-Carlo SE
   `≤ 0.0002`. If it does not, increment `L` by one and return to step 1.

`(L, B)` is written into the confirmation seal. It is never reduced afterwards for any reason, including
compute pressure; insufficient capacity is a stop.

### 17.4 The statement that must be made plainly

**Blocks and synthetic variants do not substitute for training replicates.** Adding blocks reduces only the
`σ²_block / (L·B)` term; the `σ²_lineage / L` term is irreducible without more lineages, and it is the floor of
Section 11.3. **At least six fresh paired confirmation seeds** are required, disjoint from every pilot seed. A
confirmation interval for `T_real` or `T_clock` computed over blocks rather than lineages is invalid and is not
reported, not in a table, not in a figure, and not as a robustness check.

---

## 18. Artifact contract

Every result-producing command writes an **immutable JSON** plus a **sibling `*.manifest.json`**, through the
repository overwrite guard, refusing an existing path.

**Result content.** The smallest per-cell endpoint table sufficient to rederive every point estimate, interval,
gate, `p`-value, and decision branch: one row per `(lineage, arm, block, q_D, q_A, r, world)` carrying the
**intact margin, the `AG→AC` margin, and the coupling** — both raw margins, never only their difference —
plus the cell identity ledger, counts, and exclusion reasons. Every threshold in this protocol appears in the
artifact beside the value it gates, with an explicit `passed` boolean and an explicit marker distinguishing
**fireable gate**, **forced identity**, and **recorder**.

**Manifest content.** Command; code revision and dirty state; input and output sha256 values; checkpoint hash
and, for a continued checkpoint, its parent hash and training-stream hash; parameters; seeds; device and dtype;
package versions; sample and exclusion counts; stage; status; and chronology timestamps.

**Schemas.** `gfmcd.ppm.construction.v1`, `gfmcd.ppm.mechanics.v1`, `gfmcd.ppm.fixtures.v1`,
`gfmcd.ppm.development_relocation.v1`, `gfmcd.ppm.health.v1`, `gfmcd.ppm.preflight_analysis.v1`,
`gfmcd.ppm.preflight_audit.v1`, `gfmcd.ppm.training_stream.v1`, `gfmcd.ppm.pilot_variance.v1`,
`gfmcd.ppm.confirmation_seal.v1`, `gfmcd.ppm.confirmation.v1`, `gfmcd.ppm.confirmation_audit.v1`.

**Rules.** Smoke, development, pilot, and confirmation outputs have distinct filenames and directories. A
smoke flag may never reduce `N` while retaining a confirmation filename or verdict field. Model weights,
activation captures, and fitted objects are **not committed**; if retained locally their hashes and derivation
role are recorded, and they may never be the only support for a claim. **No artifact is hand-edited after
production** — a changed interpretation is a new audit or correction artifact pointing at the immutable
result. Results are never silently overwritten.

---

## 19. The sealed confirmation

### 19.1 Co-primary directional contrasts

```
T_real  = Psi_L1  − Psi_L2       on fresh authentic modules   (role: confirmation_real)
T_clock = Omega3_L1 − Omega3_L2  on the identical-sequence designed panel (role: confirmation_designed)
```

Both are computed per lineage as paired within-lineage differences, then averaged over the `L` frozen
lineages, with the two-level resampling of Section 11.1.

### 19.2 Identification logic

The two contrasts fail in different directions, which is why both are required:

- `T_real` is a **between-checkpoint** difference on **identical sequences**, so every sequence-only confound —
  including the period-3 amplitude coupling of Section 6.4 — enters both checkpoints identically and cancels.
  What it cannot do is attribute the change to the **positional carrier**: a corpus manipulation could equally
  have changed a translation-invariant sequence-periodicity detector.
- `T_clock` is a difference of the **identical-sequence carrier intervention's** effect. By Section 9.3,
  `Omega3` is exactly zero under any `r`-invariant readout, so `T_clock` is carrier-specific by construction.
  What it cannot do is speak about **authentic modules or annotated boundary phase**: it lives entirely on the
  designed panel.

The end-to-end claim requires **both**, plus the carrier-specificity family of Section 10 and the health gates
of Section 12. Either may stand alone at the lower rung of Section 15.1 **without upgrading the other**.

### 19.3 One-shot discipline

The two confirmation panels and the `L` confirmation lineages are exposed **exactly once**, in a single
chronology recorded in the manifest. No confirmation panel is reused, re-scored under a changed rule, or
re-opened. A defect discovered **after** exposure preserves the artifact, records the exposure and the defect,
and **cannot be repaired into a confirmation on the same panel**; any rerun requires a genuinely fresh role
enumerated under a dated additive amendment.

A `REFLECTION` step is written after the confirmation artifact and its audit exist and before any claim row is
drafted: it names every gate that fired, every gate that was forced, every quantity that was recorded rather
than gated, and the strongest sentence the evidence licenses. No claim row precedes it.

---

## 20. Independent audit

Independent audits are required at four points: **construction**, **preflight authorization**, **pilot
variance**, and **confirmation result**. An audit:

- reads recorded primitive endpoints and recomputes every statistic **without importing the analysis verdict
  function**;
- verifies exact protocol-to-code threshold and endpoint coverage — every threshold in this document appears
  in the code, and every threshold in the code appears in this document;
- distinguishes **fireable gates**, **forced identities**, and **recorders**, and confirms that no forced
  identity was counted toward authorization;
- independently re-derives the handedness convention of Section 8.1 against the pure-tone fixture, and
  re-derives the token span `[393, 513)` from the nucleotide span `[392, 512)` and the `[CLS]` offset;
- checks full-family multiplicity, add-one denominators, control RMS matching, control-family size, and sign
  conventions;
- verifies role and gene disjointness, corpus non-overlap in both directions, the chromosome seal, the
  checkpoint firewall, the training-stream hash equality between paired arms, the pilot blinding key seal,
  chronology, hashes, and output immutability;
- fails if a required endpoint, arm, control, manifest field, or raw row is absent; and
- declares whether it is clean-checkout re-executable.

Interpretation requires audit `PASS`. **A recorded gate assertion that the auditor cannot rederive is an
attestation, not an independently audited pass**, and blocks confirmatory wording.

---

## 21. Anomaly, stopping, and amendment rules

1. A failed construction, mechanical, fixture, capability, health, resolution, seal, or audit gate **stops its
   dependent stage**. A still-sealed panel, corpus, or lineage is not opened.
2. An anomalous control **freezes interpretation**. Diagnose on synthetic fixtures or on the development role;
   never tune on a confirmation row.
3. A pre-exposure implementation defect may be repaired only with a dated additive amendment, a new code hash,
   a **complete rerun of the affected preflight**, and a new freeze and audit package.
4. A defect found after confirmation exposure preserves the artifact, records the exposure and the defect, and
   cannot be repaired into confirmation on the same panel (Section 19.3).
5. A failed or unrun **L3** arm stops only L3-dependent statements. It does not relax any L1/L2 gate and does
   not convert L3 into a null.
6. A failed `T_clock` does not weaken a valid `T_real`, and a failed `T_real` does not weaken a valid
   `T_clock`; evidence stops at the last rung actually passed.
7. **No sample size, dose, span, sign, register label, control family, interval type, endpoint, threshold, or
   decision rule changes after its dependent outcomes are visible.**
8. A stop for insufficient capacity, insufficient compute, or an unobtainable input is recorded as a stop. It
   is never converted into a reduced target, a relaxed threshold, or a spent seal.

---

## 22. Pre-run completion checklist

No pilot, training, or confirmation command is valid until every item is machine-checked:

- [ ] this protocol and every amendment hash-bound;
- [ ] freshness registry independently rederived from pinned sources;
- [ ] `chr8`/`chr9` absent from every enumeration, or an owner-approved amendment present;
- [ ] all five roles enumerated with global gene and paralogue-cluster disjointness;
- [ ] `k = 31` and shifts `(93, 94, 95)` rederived from the Section 6.2 rule;
- [ ] token span `[393, 513)` rederived from nucleotide span `[392, 512)` and the `[CLS]` offset;
- [ ] pure-tone handedness audit passes and rejects the off-by-one support;
- [ ] all six fixtures traverse the real analysis and decision functions;
- [ ] every control RMS-matched within `1e-10` relative; 31 hidden-axis seeds present;
- [ ] masked-probe health passes and no health probe reads the outcome channel;
- [ ] projected `MDE80 ≤ 0.05` for every endpoint in gate 7 at the frozen `(L, B)`;
- [ ] `(L, B)` frozen by the Section 17.3 rule from blinded pilot variance components;
- [ ] pilot blinding key sealed and unread;
- [ ] paired training-stream hashes agree except on offset residues;
- [ ] independent preflight audit `PASS`, issued by a party that is not the producer;
- [ ] confirmation seal written by a separate freezer on a clean tracked revision; and
- [ ] immutable output paths do not exist.

**No checked box is evidence about any effect.** The checklist certifies only that the declared question can
be asked.

---

## 23. Strongest wording by terminal outcome

Before any result from this study, the only licensed statement is:

> Whether SpliceBERT's boundary-register preference is modifiable by continued pretraining, and whether the
> learned period-3 positional carrier causally supports any such change, is untested.

On `PPM_REAL_PANEL_SHIFT_ONLY`:

> In a sealed fresh chimeric authentic-module panel at this checkpoint family, phase-decoupled continued
> pretraining shifts the model's annotated coding-boundary-phase preference relative to an edit-matched
> phase-preserving continuation. This is a behaviour-rung result; it identifies no mechanism and licenses no
> statement about the positional carrier.

On `PPM_CLOCK_SHIFT_ONLY`:

> On a designed-register panel whose token sequences are identical across carrier worlds, phase-decoupled
> continued pretraining reduces the model's response to a span-limited period-3 register intervention relative
> to an edit-matched phase-preserving continuation. The result is carrier-specific and confined to the designed
> panel; it licenses no statement about authentic modules or annotated coding-boundary phase.

On `PPM_MODIFIABILITY_SUPPORTED`:

> In this checkpoint family, phase-decoupled continued pretraining weakens the model's boundary-register
> preference on fresh authentic modules, and an identical-sequence span-limited period-3 register intervention
> shows that the learned positional carrier causally supports that change. The scope is this corpus
> manipulation, this token budget, this 120-nt span, this readout, and this assay family. It is not a statement
> about reading frame, translation, spliceosomal biochemistry, native junction usage, or any factor identity,
> and it does not establish what the original pretraining corpus caused.

On `PPM_BOUNDED_NULL`:

> At the frozen token budget, this corpus manipulation does not materially move either endpoint at the `0.05`
> scale. This bounds the effect of **this** manipulation; it does not show that the carrier is unmodifiable and
> it is not evidence that the carrier is unused.

On `REGISTER_INSTRUMENT_UNRESOLVED`: name the gate that failed and stop. **Do not write "the ridge does not
move."**
