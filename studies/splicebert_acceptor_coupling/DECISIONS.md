# Decision log — SpliceBERT acceptor coupling

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Why each experiment was designed the way it was, and what was seen before what was frozen. Numbers here
are provenance; the licensed wording lives in [`CLAIMS.md`](CLAIMS.md).

## 2026-08-05 — the study is opened, and why it is a separate study

The donor circuit's account is stated asymmetrically end to end ("an **acceptor-local** payload
delivered to a **donor-side** readout"), and no experiment in the record tests whether that asymmetry
belongs to the model or to the assay. The reversal is cheap — both readouts mask a `G`, so
`CouplingTask.margin` is reused unchanged and the swap is one position list for another.

**Separate ledger, deliberately.** Acceptor reconstruction is a different behaviour from donor
reconstruction. Filing reverse-direction rows in `splicebert_partner_selection/CLAIMS.md` would let
C1–C42 — every one of which is scoped to a masked *donor* — silently acquire a second meaning. `AC*`
rows here; `C*` rows stay where they are.

**One prior operator result is folded in rather than re-derived.** `AR1`'s `DONA` arm measured
ρ ≈ 0.62 for every donor alt with `z[1] == "T"`, motivating R1 to treat edit size as a factor rather
than a nuisance. An early version of this decision log also imported a numerical comparison and verdict
from the now-archived exon-edge compatibility prose. `R45` superseded that source record, so the imported
comparison is withdrawn. Only `C43` and `C44` survive that consolidation: the former is a bounded
masked-donor phase result in a chimeric crossed lattice and the latter is instrument-unresolved. Neither
is evidence for native masked-acceptor operator retention or reciprocal symmetry.

### Pilot 1 — n = 200 val, 180 genes. DIAGNOSTIC, endpoints seen.

| quantity | value |
|---|---|
| clean margin, donor / acceptor | `+4.373` / `+3.632` |
| forward `Y_D` (acceptor `AG→CC`, read donor) | `+0.9224 [+0.7876, +1.0711]` |
| reverse `Y_A` (donor `GT→AC`, read acceptor) | `+0.9030 [+0.7600, +1.0491]` |
| reverse `Y_A` (donor `GT→AT`, one substitution) | `+0.5483 [+0.4594, +0.6443]` |
| non-partner `GT` decoy | `+0.0090 [-0.0078, +0.0260]` |
| other-acceptor readout | `+0.0434 [+0.0023, +0.0947]` |

Established that the swap runs and that the forward anchor reproduces. **Its other-acceptor control was
wrong, not merely noisy:** it took `real_acc[0]` with no distance matching. An audit of 1,200 val exons
put those sites at a median offset of `+157` from the donor against the partner's median gap of `95`,
with **36.9% upstream of the donor**. `+0.0434` is superseded and must not be quoted.

The fix was to stop inventing a control and import C2's. `run_sealed_test.py` builds two, both from
*spurious* distance-matched motifs rather than real annotated sites: the same corruption read at a decoy
position (`matched_decoy_donor`), and a decoy corruption read at the real position
(`matched_decoy_acceptor_null`). The mirror is `matched_decoy_acceptor` for the readout arm and
`matched_decoy_donor` for the corruption arm.

### Pilot 2 — n = 1000 val → 965 paired, 623 genes. DIAGNOSTIC, endpoints seen.

Zero planted motifs across 965 constructs.

| quantity | value |
|---|---|
| clean margin, donor / acceptor / decoy-`AG` | `+4.4813` / `+3.6016` / `+0.1895` |
| forward `Y_D` | `+0.8976 [+0.8361, +0.9677]` |
| reverse `Y_A` | `+0.9749 [+0.9005, +1.0472]` |
| control — same corruption, decoy-acceptor readout | `-0.0057 [-0.0202, +0.0069]` |
| control — decoy-`GT` corruption, own acceptor readout | `+0.0284 [+0.0100, +0.0509]` |
| paired difference vs readout control | `+0.9806 [+0.9040, +1.0558]` |
| paired difference vs partner control | `+0.9465 [+0.8662, +1.0236]` |
| selectivity ratio | ~34× `[22.9, 37.3]` |

**The operator prediction was stated before the sweep was read, and it landed.** The split falls on
`z[1]` and on nothing else:

| `z[1]` | alts | ρ vs `AC` |
|---|---|---|
| `C` | `AC` 1.000, `TC` 0.999, `CC` 0.984 | ≈1.00 |
| `G` | `TG` 0.979, `CG` 0.977 | ≈0.98 |
| `A` | `AA` 1.021 | ≈1.02 |
| **`T`** | `CT` 0.690, `AT` 0.679, `TT` 0.678 | **≈0.68** |

AR1's `DONA` reached ρ ≈ 0.62 on 190 examples drawn entirely from the 19% of donors lacking the
consensus `G`. This is ρ ≈ 0.68 on 965 across the full cohort. **The dose-response is real and survives
leaving the selected subcohort** — it was not an artifact of R44's admissibility filter.

### The decision rule was corrected after pilot 2, and that is disclosed in the prereg

The drafted rule required each control arm's own CI to lie inside `±SESOI` as a second conjunct of `GO`.
Pilot 2's decoy-`GT` control returned `[+0.0100, +0.0509]` — **outside by 0.0009**, on an effect of
`+0.975`. The conjunct was removed before freezing because it was mis-specified, not because it was
inconvenient: (i) it is stricter than the standard the forward direction was held to — C2 establishes
partner-specificity from *paired differences* and states no equivalence test on control magnitudes, so
gating the mirror on a criterion its original never faced defeats the comparison the study exists to
make; and (ii) the classes were not mutually exclusive — a control could fail the `GO` conjunct while
not triggering `NO_GO`, landing in `INSTRUMENT_UNRESOLVED` no matter how large the paired difference.
IC2's freeze-audit recorded this same defect shape as a must-fix.

**The cost is not mitigated:** the rule was touched after the number it governs was seen, and under the
drafted rule this gate does not return `GO`. The artifact records `verdict_under_drafted_rule` beside
the frozen verdict so both readings stay recoverable, and prereg §7.1 states it. A reader who holds
that a post-hoc threshold is void should read the drafted outcome.

### R1 settled run — n = 970 paired, 648 genes. All eight memorials PASS.

Executed by the committed runner after freezing. **The sampling tag differs from both pilots, so
`load_examples` drew an independent val cohort — these are a replication, not a re-report.**

| endpoint | value |
|---|---|
| `Y_A` reverse | `+1.0200 [+0.9472, +1.0916]` |
| `Y_D` forward anchor | `+0.9436` (AR1 val reference `+0.9909`, tol `±0.15`) |
| control — same corruption, decoy-acceptor readout | `-0.0181 [-0.0327, -0.0063]` |
| control — decoy-`GT` corruption, own acceptor readout | `+0.0129 [-0.0005, +0.0270]` |
| paired difference vs readout | `+1.0381 [+0.9628, +1.1113]` |
| paired difference vs partner | `+1.0071 [+0.9334, +1.0816]` |
| selectivity ratio | `19.9 [15.7, 25.2]` |
| clean margins — donor / acceptor / decoy-`AG` | `+4.5605` / `+3.6682` / `+0.1786` |

**Verdict `INSTRUMENT_UNRESOLVED`**, from §6.3's `MDE80` gate (`0.106` on the paired differences against
a `0.05` band). Logged as a specification defect in [`CORRECTIONS.md`](CORRECTIONS.md) `ACR1` and
**not repaired post-hoc**; the same gate on the control arms — the quantity that actually needs power —
reads `0.019` and `0.029`.

**The §7.1 rule correction turned out immaterial.** The decoy-`GT` control landed inside `±SESOI` on this
cohort, so `verdict_under_drafted_rule` equals `verdict`. The disclosure stands anyway.

**The operator prediction replicated on the independent draw**, with no CI overlap between the groups:
`T` retained at `d+1` → `AT` 0.673, `CT` 0.675, `TT` 0.653; `T` rewritten → `AC` 1.000, `CC` 0.996,
`TC` 0.996, `CG` 0.988, `TG` 0.963, `AA` 1.018.

**The selection audit produced a result of its own.** On the `CC`-admissible stratum — 17.7% of sites,
and the only sites AR1's `DONA` arm could measure — `Y_A = +0.8294` against `+1.0200` on the full
cohort. `DONA` was **understating the reverse coupling by ~19%**, because R44's admissibility filter
restricted it to donors lacking the 5'SS consensus `G`. That is `AC3`.

### 2026-08-05 (later) — an adversarial design review found three things the runs could not

A four-agent reconnaissance of the record, with adversarial verification of every finding (6 of 14
survived), returned three results that change what is written above. Each was re-verified by hand
before being acted on; the agents' unverified numbers are not used.

**1. `L3H1` transfers, and the reverse direction's asymmetry is in DELIVERY, not in the reader.** The
2026-07-19 withdrawal of the atlas's acceptor rows was about the A/B/C/D matrix. A **separate**
motif-clean screen, `results/splicebert_head_context_atlas/head_screen_clean_alt/` (400 shards,
`destroy_alt_mode = MOTIF_CLEAN_ALT=AC` — R1's own operator), was never withdrawn. Recomputed here from
all 400 shards with gene-clustered bootstraps:

| head | acceptor-target B | donor-target B | ratio |
|---|---|---|---|
| **L3H1** | **+0.1825 [+0.1175, +0.2503]**, 82% of behaviour | +0.0537, 71% | 3.40× |
| L4H7 / L0H7 / L4H11 / L4H9 | +0.0259 / +0.0199 / +0.0279 / +0.0271 | +0.0012 / +0.0016 / +0.0028 / +0.0040 | 21.8× / 12.4× / 9.9× / 6.8× |
| **L5H11** | **+0.0010** (CI covers 0) | +0.0098 | **0.11×** |
| L4H0 / L4H2 | +0.0036 / +0.0083 | +0.0067 / +0.0088 | 0.54× / 0.95× |

L3H1 is **rank 1 of 96 at the acceptor readout, by 6.5× over rank 2**, and carries a larger share of
behaviour there than at donor targets. The forward transporter L5H11 collapses; a different L4 band
rises. **Shared reader, swapped transporters** — not "the same circuit backwards".

**This may shortlist and may not be cited.** The artifact is **gitignored** (`.gitignore:50`) and
untracked; `planted_motif_enforced = false`; and it is the train-only alt-3′ss competition cohort at
behaviour `+0.2226` against R1b's val `+0.9763`, a 4.4× different regime. It is also a **noising**
estimand, which under-ranks address-side components by ~45× (forward L2H15: `+0.0115` noising in
`donor_heads.json` `sanity` vs `+0.5146` mean-ablation in `development/ablation_scan.json`) — so it says
nothing about the reverse address.

**2. A withdrawn null is still live in a canonical artifact.**
`results/splicebert_head_context_atlas/head_effect_reanalysis_train_only_v1.json` — registered
`canonical` at `results/MANIFEST.csv:941` — still carries the coordinate-invalid `directional_B`
acceptor values (L3H1 `−0.001037`, L5H11 `−0.002305`, L4H2 `−0.001451`, L0H7 `−0.000853`). They read as
a clean reciprocal null. This is the most likely reason a reader — including this study's own first
pass — concludes that L3H1 does not transfer. **It belongs to the Atlas study and needs an R-number
there**; it is recorded here because this study found it.

**3. The "less selective" claim below was wrong, and is retracted.** It compared incompatible
statistics. `sealed_test.json` reports forward `selectivity_ratio = 19.5996 [11.98, 37.24]` as a clipped
ratio-of-means; the "~125×" came from the naive mean ratio (`0.8725 / 0.00728 = 119.9`). Like-for-like,
the reverse is `18.16 [14.64, 22.69]` — **indistinguishable from the forward**. A residual caveat
survives even so: the forward statistic divides by the **readout** control and this study's divides by
the **corruption** control, so the two are still not the same estimand and no "as selective as" claim
should rest on them until one definition is fixed in one place and both directions are recomputed.

**Also refuted here: AC2's mechanism clause.** See [`CORRECTIONS.md`](CORRECTIONS.md) `ACR4` and the R2
arm. The nine-alt panel contained two of three cells; `GA`/`GG` supply the third and show the reverse
coupling responds to **either** base of the `GT`, sub-additively, with the `T` contributing more.

### R3 settled run — 974 exons / 645 genes, both readouts, 96 heads, all nine memorials PASS

> **Historical artifact output; inferential reading withdrawn.** `ACR9` shows that the specificity,
> permutation, planted-positive, and equivalence paths did not implement the frozen familywise rule, and
> `ACR5` records the missing noising estimand. The table below preserves the raw mean-ablation values and
> artifact labels as provenance. “All nine memorials pass,” the 73/17/4/2 partition, both rank
> identifications, and direction-specific labels are not live claims. `ACR6` had already withdrawn
> ordering below rank 1; `ACR9` supersedes that narrower disposition.

`Y_D = +0.9792`, `Y_A = +0.9763` on identical introns — the two directions differ by `0.003` at the
behaviour level, so nothing below is explicable by one being a bigger effect. Pseudo-couplings
`+0.0264` / `+0.0147`.

**Partition: 73 `SHARED`, 17 `UNRESOLVED`, 4 `FORWARD_SPECIFIC`, 2 `REVERSE_SPECIFIC`.**

| head | `frac_D` | `frac_A` | `Δ` | class |
|---|---|---|---|---|
| **L3H1** | +0.8210 | +0.8851 | +0.0640 [−0.0039, +0.1318] | UNRESOLVED — **rank 1 in both** |
| **L2H15** | +0.4731 | +0.5045 | +0.0314 [−0.0230, +0.0862] | UNRESOLVED — rank 2 in both |
| L1H12 | +0.2114 | +0.2610 | +0.0497 [+0.0101, +0.0863] | UNRESOLVED |
| L2H4 | +0.0896 | −0.0347 | −0.1243 | FORWARD_SPECIFIC |
| L4H2 | +0.0832 | −0.0126 | −0.0959 | FORWARD_SPECIFIC |
| L5H11 | +0.0809 | −0.0063 | −0.0872 | FORWARD_SPECIFIC |
| L1H8 | +0.0363 | −0.0408 | −0.0771 | FORWARD_SPECIFIC |
| **L2H5** | −0.0695 | +0.1168 | **+0.1863** | REVERSE_SPECIFIC — largest of any component |
| L4H12 | −0.0814 | +0.0109 | +0.0923 | REVERSE_SPECIFIC |
| L4H0 | +0.0414 | +0.0209 | −0.0205 | SHARED |

**The mean-ablation choice was decisive, exactly as prereg §2 argued.** `L2H15` reads `+0.473`/`+0.505`
here against `+0.0183` in the noising-based shortlist. Had R3 used the shortlist's estimand it would
have concluded the addresser was irrelevant **in both directions**. The 45× forward discrepancy
generalises to the reverse.

**Three of the four pre-stated expectations were wrong, and the fourth was only half right.**
`L5H11 FORWARD_SPECIFIC` held. `L3H1 SHARED` did **not** — it is rank 1 in both but `UNRESOLVED`, and
`UNRESOLVED` is explicitly not `SHARED`. `L4H11`/`L4H9`/`L4H7 REVERSE_SPECIFIC` did **not** — they came
back `UNRESOLVED`/`UNRESOLVED`/`SHARED` (`Δ` `+0.048`/`+0.054`/`+0.015`), reverse-leaning in sign but
none material. **The shortlist's direction transferred; its magnitudes did not** — which is what a
4.4×-weaker train cohort under a different estimand should be expected to do, and is a finding about
the shortlist rather than about the mechanism.

**Historical candidate note (not a result after `ACR9`): `L2H5`.** It sits in the same layer as the forward addresser, is
`REVERSE_SPECIFIC` with the largest `|Δ|` in the scan, and its two arms have **opposite signs**:
ablating it *raises* forward coupling (`frac_D = −0.070`) and *lowers* reverse coupling
(`frac_A = +0.117`). The same shape appears at `L4H12` (`−0.081` / `+0.011`). Nothing in the record
names either head. Suppression of one direction while supporting the other is not something the
forward account predicts, and it is not explained here.

**Health, interpreted rather than asserted.** Every component passes the `0.5` floor at `0.52–0.59`,
and the **unablated** baseline is `0.529` forward / `0.585` reverse, with single-head ablation moving it
by `≤0.003`. So the range is the model's intrinsic masked-probe accuracy (~2× the 0.25 chance level),
not damage: "mechanism, not destruction" holds. **But the gate has only `0.029` of headroom above its
floor**, so it fires only on near-total destruction. It is a coarse gate and should not be quoted as a
tight one.

### Two weaknesses recorded now, not after someone finds them

1. **The readout-specificity control sits near floor.** It reads at a spurious decoy `AG` whose clean
   margin is `+0.19` against a real acceptor's `+3.60`. A readout with little signal cannot lose much,
   so `-0.0057` is a weak control by construction. This is **inherited from C2's own idiom** — the
   forward direction's other-donor control reads at a spurious `GT` — and so applies to both
   directions equally. It is a limitation of the comparison, not of the reversal.
2. ~~**The reverse direction is measurably less selective than the forward.** Its non-partner control is
   `+0.028` where C2's is `+0.007`, about 4×; selectivity ~34× against the forward's ~125×.~~
   **RETRACTED 2026-08-05 — both halves were wrong.** The ratio comparison mixed a clipped
   ratio-of-means (`sealed_test.json`: `19.5996 [11.98, 37.24]`) with a naive mean ratio
   (`0.8725/0.00728 = 119.9`, the source of "~125×"); like-for-like the reverse is
   `18.16 [14.64, 22.69]`, **indistinguishable from the forward**. And the `+0.028` control was pilot 2's
   outlier, which `ACR3` identifies: R1b's three draws give `0.0147 / 0.0156 / 0.0144`, SD `0.0006` —
   about **2×** C2's `0.00728`, not 4×. See the 2026-08-05 (later) entry above for the surviving caveat,
   which is that the two directions' selectivity statistics still divide by different control arms.

## 2026-08-08 — the BP arc: what it costs to leave the short-intron band

An exploratory probe asked whether the acceptor-local payload (`C30`, semantics UNRESOLVED, bounded
by `R22`) is branchpoint / polypyrimidine content, motivated by a verified literature window —
branchpoints in U2-type **constitutive** human introns sit a median 28 nt upstream of the 3′ site,
80 % within −49…−20 (Pineda & Bradley 2018, *Genes Dev* 32:577; corroborated at median 25 nt,
90 % within 19–37 nt by Mercer 2015, *Genome Res* 25:290).

**No prereg was frozen and nothing here is ledger-eligible.** No `AC*` row is created or changed.
Runners and artifacts live outside the tree; only the conclusions are recorded here.

### Two designs were killed before they consumed confirmatory compute

1. **The window does not fit the cohort.** `circuit.eligible` bounds the donor→partner-acceptor span
   to `GAP_MIN, GAP_MAX = 60, 150` nt; observed val median **95** (5th pct 74, 90th 135). A tiling
   that reaches −199 fits **0 of 1200** examples. On a 95 nt intron the literature window is a third
   of the whole intron, so "branchpoint zone" and "the intron" are not spatially separable and the
   contrast is not identifiable on this cohort even in principle.

2. **A `matched_decoy_acceptor`-anchored positional profile manufactures its own positive.** The
   helper matches distance to the **donor**, which *arithmetically* forces the decoy within `tol`
   (40 nt) of the acceptor. A profile anchored at the decoy is therefore the real profile shifted by
   δ < one window, so the anchor difference `D(j) = Δ(x_j) − Δ(x_j + δ)` is a discrete **derivative**
   of the decay curve. Under any smooth convex decay with **zero** branchpoint content it yields
   `D(0) > 0` and `D(0) > mean(D(1..5))` — a false localisation.
   **This does not affect R1/R1b/R3/R4 or the GTEx arm**, which use the helper for point comparisons
   at a single decoy location, not as an anchor for a positional profile. Recorded so nobody builds
   a profile on it later.

### The coupling is a short-intron phenomenon, and that is an intervention result

`README.md` already scopes this study to "canonical internal short introns". What was not recorded
is the size of the cliff. One operator (`AG → CC`, the `AC1`/`C1` operator), one code path, only the
span varying, n = 250 per bin, gene-clustered 95 % CI, val:

| span (nt) | median | coupling |
|---|---|---|
| **60–150** | 96 | **+1.017 [0.887, 1.158]** — reproduces `C1` (+0.872) and `AC1`'s forward anchor (+0.968) |
| 150–250 | 191 | +0.336 [0.271, 0.405] |
| 250–400 | 319 | +0.065 [0.039, 0.099] |
| 400–700 | 501 | +0.147 [0.115, 0.183] |
| 700–1500 | 739 | +0.116 [0.081, 0.156] |

That table **selects** on span, which confounds span with the partner acceptor's absolute position,
the surviving 3′ context and long-intron composition. The layout makes this acute:
`acc_win = randint(100, 300)` and `min_context = 100` is enforced after the exon's OWN donor and
**never** after the partner acceptor, so a long span pushes the partner toward the window edge and
only survives when the donor is early. SpliceBERT has **absolute** learned position embeddings, so
absolute position is not a neutral covariate.

So the span was **intervened on** instead. One base cohort (60–150 nt), two manipulations that both
move the partner acceptor right by `k` with the same 3′ tail; filler is a dinucleotide-preserving
shuffle of the example's own intron. n = 300 / 258 genes:

| k | INSERT (span + k) | SHIFT (span fixed) | partner acceptor abs. position |
|---|---|---|---|
| 0 | +0.864 | +0.864 | 410 |
| 100 | +0.420 | +0.655 | 510 |
| 200 | +0.097 | +0.625 | 610 |
| 400 | **+0.155** | **+0.752** | 810 |

`k = 0` is identical by construction in both arms — the internal consistency check. At `k = 400`
the arms differ **4.8×** at matched position and matched tail. **Span effect −82 %; position/edge
effect −13 to −28 %.** The decay is a real span dependence, not a windowing artifact.

Secondary, and not what was being looked for: the donor's **own clean margin** degrades with span
(4.63 → 3.66) but not with position (4.63 → 4.47). The model's confidence that a donor is a donor
depends on a partner acceptor being nearby.

### A positionally specific branchpoint reader exists, and it is small

With identification moved from position to **content at fixed coordinates**: at each 30 nt window,
dinucleotide-preserving shuffle, then plant either `CTAAC` (yUnAy consensus) or a
composition-matched scramble (`ACACT`/`CAACT`/`TCACA`) at the same coordinates. Both arms edit the
same bases of the same molecule and differ only in the **order of five nucleotides**, so geometry,
composition and window identity cancel exactly. No planted motif contains `AG` or `GT`, and the
dinucleotide-preserving shuffle holds `AG`/`GT` counts fixed — R44 planting is zero by construction
rather than by threshold. n = 1000 / 750 genes, val, span ≥ 250, all nine memorials PASS,
sensitivity gate +0.117 [+0.100, +0.136].

Recovery is **+0.00850 [+0.00670, +0.01031]** at −49…−19 (p = 0.0005, Holm ×6 = **0.0030**) and
straddles zero at all five other windows (Holm 1.000 each). Contrast `j0 − mean(j1..5)` =
+0.00849 [+0.00663, +0.01034]. Destroying *order* shows no gradient across windows; restoring a
*specific motif* localises — a motif-specific reader, not general order-sensitivity.

**Four things bound it.** (i) It is 7.3 % of the +0.117 residual coupling it sits inside — 93 % is
something else. (ii) +0.0085 against a clean margin of 3.34 is **~6× below this study's SESOI of
0.05**: statistically unambiguous, practically small. (iii) It is measured at span ≥ 250 nt, i.e.
**outside** the 60–150 nt band where `C1`/`AC1`/`C30` live and which carries 5× the coupling; that
band cannot be probed this way because the window does not fit. (iv) One draw, one seed set —
`R22` requires multiple seeded draws plus two-way dyadic inference before any effect-existence
claim.

### A statistics defect, found and corrected

The first pass reported `p = 2·min(frac(v>0), frac(v<0))` over per-example values — a **sign test on
examples**, not a test of the mean. It returned p = 0.808 for a mean ~9 SE from zero, and the Holm
column was a correction over garbage. The frozen decision rule tested CI-excludes-zero and the CIs
were always correct, so **no verdict depended on it**. It was replaced by a two-sided
percentile-bootstrap p drawn from the *same* gene-clustered resampling that produces the CI, with a
new memorial asserting the mirrored CI is bit-identical (< 1e-12) to `circuit.gboot`'s. The re-run
reproduced every mean and CI bit-identically; only the p column changed.

### One scope tension, flagged rather than resolved

`README.md` states this study is "model-internal throughout — **no external biological assay enters
this study at any point.**" That still holds literally: no assay was run. But the BP arc puts an
externally-derived biological motif (`CTAAC`) into the *stimulus*, and takes its window from the
branchpoint literature. That is external biological content entering the input, which the scope
sentence was probably written to exclude. **Not resolved here.** Either the arc belongs in a
different study, or the scope sentence needs amending — a reviewer's call, not a cleanup pass's.

## 2026-08-11 — corrected closeout instruments: what is ready, what remains unrun

No model was run in this closeout. The work establishes executable, auditable successor designs; it
does not yet establish R3b cohort capacity, creates no `AC*` row, and does not revisit any historical model outcome.
The controlling status correction is `ACR13`.

### Sri remains a historical hypothesis source, not a confirmation cohort

Sri's branch and the paired-boundary program remain separate. Sri repeatedly used the same 200 GUE
acceptor-test examples with a joint masked-`AG` local-context endpoint. The current study uses a
single-`G` paired-donor endpoint. The branch's missing raw rows and gene-bound identifiers mean its
sample cannot be intersected with either corrected successor cohort. Accordingly, the four Sri-only
R3b registry entries (`L5H8`, `L3H5`, `L0H13`, `L0H14`) are frozen as exploratory hypotheses from
commit `6a136390`, and nothing is described as a gene-disjoint replication of Sri.

### R1c closes the behavior-instrument defects, not the behavior claim

R1c is deliberately behavior-only. It replaces R1/R1b's near-floor primary spurious-`AG` readout with
a live annotated exon-mate acceptor, keeps the legacy readout descriptive, applies one sign per gene,
and binds the complete endpoint-scale-normalized decision path and runtime contract. The split is literal:

- preflight: 400 outcome-blind rows on validation chr11/16/19;
- target: 400 outcome-blind rows from the R1c endpoint-outcome-held-out chr8/9 cohort (`ACR18`).

The target is unreachable while the protocol is draft or either reviewed pin is absent, and requires a
fresh target-bindable preflight after freezing. No such preflight has run. Therefore R1c's strongest
status is **implemented draft behavior-confirmation instrument**. It supplies no sealed result and
does not promote AC1 beyond its bounded validation-gate wording.

### R3b closes the component-instrument defects, not the component claim

R3b starts by making the old exclusion population immutable: historical R3 must reconstruct to 974
rows / 645 genes with ordered example-ID SHA-256
`161dff038d747dbaa7b91a96d24b34d45f55de5482b5140410641354e46d2a8e`, after which every represented
gene is excluded. Current pseudo controls are distance-calipered at 40 nt in both directions and are
called non-partner sequence motifs because annotated overlap is recorded, not assumed absent.

The runner now constructs the corrected outcome-blind census and requires a fixed first-500 prefix with
at least 30 genes in each bucket. A temporary check was not published with an immutable artifact and
manifest, so it cannot establish the exact counts or capacity. The first conformant preflight must bind
that audit to the exact source and transformation bytes. Both planned buckets draw only from validation
chr11/16/19:

- bucket 0: development and future target-bindable preflight;
- bucket 1: held-out within-validation target.

This is not the R1c target cohort, and it scores none of those masked-acceptor target outcomes. R3b now implements matched-size
head/MLP families, whole-gene endpoint-scale-normalized simultaneous inference, within-example token-mean ablation,
same-row corrupt-to-clean noising, health-aware complete-path null/plant calibration, and exact
two-pin preflight binding. None of those paths has been run on the model. Its strongest status is
**implemented draft focused-component instrument; cohort capacity unestablished**. AC4's raw values
remain candidate generators; its nominal partition, ranks, and direction labels remain retracted or
instrument-unresolved exactly as recorded in `ACR9`/`ACR11`.

### Promotion boundary

R1c and R3b require separate freezes and separate passing preflights. R1c alone can spend the sealed
chromosome target and at most establish the bounded reciprocal behavior. R3b's within-validation
bucket target can at most nominate intervention-specific component necessity under its two estimands.
Neither program by itself licenses address, payload, write, delivery, mediation, semantic transfer,
generative faithfulness, biological correspondence, or a cellular splicing mechanism.

## 2026-08-15 — R1c is redesigned around the masked acceptor, not a mirrored donor assay

The first complete R1c development run exposed two instrument mistakes: the live upstream exon-mate
acceptor responded to the paired-donor edit (`+0.3180`), so requiring it to be equivalent was invalid;
and the effect-bearing raw-outcome plant returned `0/200`. `ACR14` records the artifact, hashes, exact
diagnostic values, and non-evidence boundary. The failure is preserved; it is not converted into a
behavior result.

The pre-freeze successor asks the acceptor-only question as a readout-by-edit 2x2:

| | real paired donor `D` | matched non-partner donor `N` |
|---|---|---|
| downstream intron-mate acceptor `P` | `Y_PD` | `Y_PN` |
| upstream exon-mate acceptor `X` | `Y_XD` | `Y_XN` |

The sole simultaneous family is `[Y_PD, Y_PN, S_P, I]`, where `S_P = Y_PD-Y_PN` and
`I = S_P-(Y_XD-Y_XN)`. The core claim requires positive `Y_PD`, equivalent `Y_PN`, and positive `S_P`.
The stronger downstream-preference claim adds positive `I`. `Y_XD`, `Y_XN`, and `S_X` remain
descriptive and are not required equivalent. Both paths therefore use the exon-mate response without
mistaking its existence for instrument failure. The masked-donor forward anchor is removed entirely;
clean margins at both acceptors and non-site MLM health are the positive controls.

`I` is restricted to outcome-blind dyads with
`abs(exon_length - intron_length_down) <= 20 nt`, chosen before `Y_XN` or any target outcome was scored.
A finite-pool full-GTF construction audit found 425 eligible rows / 339 genes among 1,501
comparator-complete rows; the first 400 retain 321 genes. The production implementation now scans the
entire already-allowed chromosome split before applying the same fixed filter, so capacity cannot
depend on an arbitrary prescreen. It preserves the v2 deterministic cohort ordering while binding the
new construction, exhaustive census, and ordered hashes. This audit is development-only and says
nothing about chr8/9 target capacity.

The plant was also fixed before `Y_XN` at raw-cell vector `[0.40, 0, 0, 0]`, which algebraically maps
to primary `[0.40, 0, 0.40, 0.40]`. Every outer null/plant trial subtracts each cell's observed mean,
applies one shared sign per whole gene across the four cell residuals, and derives `S_P`/`I` afterward.
The conservative null still counts any material primary endpoint as a false positive; core and
stronger plant power are scored separately. With 200 trials the exact one-sided 95% Clopper--Pearson
boundaries are frozen: at most 4 null positives and at least 188 successes for each planted path.

The post-redesign path has now run once as a draft development preflight on 400 validation rows / 321
genes. It returned `PREFLIGHT_GO`: `Y_PD=+1.1947 [+1.0497,+1.3397]`,
`Y_PN=-0.0033 [-0.0193,+0.0127]`, `S_P=+1.1980 [+1.0531,+1.3428]`, and
`I=+0.9213 [+0.7488,+1.0938]`; the previously missing `Y_XN` cell was `+0.0122`. All gates passed,
including 0/200 conservative null positives and exact-bound power for both planted paths. `ACR14`
records the artifact hashes and scope.

This is still a draft, dirty-worktree development result. Neither seal pin is installed and the R1c
masked-acceptor target outcomes remain unscored, so it is not a target-bindable preflight and creates no `AC*` claim. A later sealed
core result could establish bounded paired-donor dependence at the masked downstream acceptor; only the
stronger path could add downstream-over-exon-mate preference. Neither behavior path licenses a component
or an address/payload/write/delivery mechanism.

## 2026-08-15 — v3 is retained as development history; v4 is the only forward R1c instrument

The preceding entry records the v3 decision as it was made and the immutable artifact facts remain
unchanged. A later inferential audit found that v3 joined SESOI-boundary interval classifications to a
raw randomisation p-value at zero. It did not directly test material positivity at `+0.05` or `Y_PN`
equivalence at both `-0.05` and `+0.05`. `ACR15` therefore withdraws the v3 artifact as valid
target-binding evidence without deleting its `PREFLIGHT_GO` self-report, hashes, cohort, estimates, or
historical calibration.

A diagnostic replay of the stored sufficient rows, with no new model scoring, used explicit
whole-gene boundary tests, `Y_PN` TOST, path IUT, Holm over the two nested paths, and strict
simultaneous-CI concordance. At 2,000 shared sign draws all constituent boundary edges were `1/2001`,
both adjusted path p-values were `2/2001`, and both paths were declared. The strict intervals were
`Y_PD [1.0422,1.3472]`, `Y_PN [-0.0194,0.0128]`, `S_P [1.0458,1.3501]`, and
`I [0.7418,1.1008]`. These are supportive diagnostics on already-observed development rows, not a new
preflight, replication, or claim.

The exact-path calibration first failed narrowly in its registered 200-trial window: the core worst
surface was `5/200` with one-sided 95% Clopper--Pearson upper bound `0.05184`; the stronger worst
surface was `4/200`, and the core/stronger plants succeeded `200/200` and `196/200`. The failure is
preserved. Extending the same deterministic namespaces to 1,000 trials gave worst surfaces `10/1000`
and `9/1000` (upper bounds `0.01690` and `0.01565`) and plant successes `1000/1000` and `982/1000`
(lower bounds `0.9970` and `0.9734`). The design response was to improve audit precision, not to alter
scientific alpha, SESOI, seed, or observed-data thresholds.

A final statistics-only replay then exercised the hardened production implementation at 4,000 inner
draws and 1,000 outer trials on the same retained v3 primitive rows. It gave core/stronger worst
boundary counts `12/1000` and `18/1000` (upper bounds `0.01937` and `0.02658`) and plant successes
`998/1000` and `983/1000` (lower bounds `0.99372` and `0.97461`), with compact trial-record SHA-256
`ed620e46dc07250912678c16740de4204efabe0ef3791c486aef1ac3a1dbc2b9`. No model was loaded and no
artifact was written; this is implementation verification on observed development residuals, not a
fresh or target-binding v4 preflight.

R1c v4 is now the sole forward behavior instrument. It binds arbitrary-boundary whole-gene sign tests,
TOST/IUT/Holm, strict-CI concordance, 4,000 shared inner draws, the expanded 1,000-trial audit, and an
exact reviewed preflight-artifact pin in addition to the protocol and instrument pins. It remains
**DRAFT, unpinned, and unrun on the model**. No fresh v4 preflight or target exists; v3 cannot be
grandfathered into the v4 target seal. The R1c target outcomes remain unscored (`ACR18`).

### The component and mechanism path is acceptor-only

R3c replaces R3b in the current forward path. R3b remains a never-frozen, unrun design record; its
reciprocal donor endpoint and head+MLP family do not answer the revised acceptor-only question. R3c
uses only the masked downstream-acceptor `G` margin, excludes historical R3 genes, fixes ten heads and
two intervention modes, and tests a 62-row family through seven named paths with independent live-hook
controls. The implementation and hard target seal exist. No conformant smoke, bucket-0 preflight, or
bucket-1 target has run. A disclosed two-row developer execution exercised hook mechanics only; it did
not satisfy the cohort, family, health, calibration, or decision gates and is not evidence. R3c does
not score the R1c masked-acceptor target outcomes on chr8/9.

R5a, R5b, and R5c are conditional future method drafts, not completed experiments. They respectively
specify paired-donor address/value interventions, a cross-fitted acceptor-side residual state, and
`L3H1` write plus conditional late-head delivery tests. They have no runner, frozen hashes, preflight,
target artifact, or result. Their methods must be frozen before the first shared evaluation outcome;
failure of an upstream gate closes the dependent stage rather than permitting a post-outcome component,
rank, or route substitution.

## 2026-08-15 — R3c v2.1 completes the held-out acceptor-only component target, but does not open the R5 chain

Two predecessor failures remain part of the decision record. R3c v1's development preflight observed
`L2H15_mean_component` health at `31/64` and suppressed its plant, but its target rule could have revived
the path from bucket-1 health. No bucket-1 outcome was opened under v1; `ACR16` withdraws that
authorization interpretation while preserving the v1 artifact SHA-256
`b4d9172b911588f7ba83e7441564851b0111763e3c0ffea5931403dc27d3f913`. R3c v2 made health
authorization monotone and its clean `r3c_acceptor_head_preflight_v2.json`
(`r3c_acceptor_head_preflight_v2.manifest.json`)
passed, but an invocation missing `--preflight-artifact` inspected the supplied model path before it
failed. No bucket-1 outcome was opened under v2. `ACR17` preserves that authorization-order defect; the
changed runner bytes required a fresh preflight rather than reusing v2.

The v2.1 early guard runs before any caller-supplied model, GTF, FASTA, output, cohort, or preflight path
is inspected. A fresh clean [v2.1 preflight](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.json)
([manifest](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.manifest.json))
then authorized the gene-disjoint [r3c_acceptor_head_target_v2_1.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json)
([r3c_acceptor_head_target_v2_1.manifest.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.manifest.json);
[r3c_acceptor_head_target_v2_1.audit.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json)).
The artifact / manifest SHA-256 pairs are
`c0c9306fe894ce73bfd981a14c85e2aec6660da33f01d9716cee4327b3ff2d3e` /
`4a53f96b365c7bed227c8e0c995cee65a022620994e8898bab988b354fce41bb` for preflight and
`ebd4bfd8c10cc0ccb218b3f1ceca634605903d8afbb5617de312087100627cc6` /
`dc644901224227f03e683f7416587dccf3eedd700052b211089242b9b8d69cbe` for target.
The audit SHA-256 is `0363d02dced48d0d1a8ba56f1862be4a0cbb0b82eb508f9aeb87b8c61b05edb3`;
it reports `AUDIT_PASS_MIXED_SCIENTIFIC_OUTCOME` after two exact replays without model rescoring or
cohort reopening.

The target stayed on validation chr11/16/19, used bucket 1 (500 rows / 304 genes), and did not score
the R1c masked-acceptor target outcomes on chr8/9. Its native paired-donor effect was `+0.9749`, matched pseudo-donor effect `+0.0232`,
and clean masked-acceptor margin `+3.5515`. It constructs exactly one masked acceptor `G` endpoint and
no masked-donor endpoint. The seven frozen path dispositions are:

| Path | Disposition | Decision |
|---|---|---|
| `L3H1_reused_component` | `INCONCLUSIVE_METHOD_DISAGREEMENT` | Holm-adjusted `p=0.02724` passes, but pseudo equivalence is not certified by the strict simultaneous intervals under either intervention. No reuse claim. |
| `L2H15_mean_component` | `HEALTH_FAILED` | Current health passes `34/64`, but the reviewed v1 `31/64` failure keeps it outside the monotone authorized set. Numerical endpoints are diagnostic only. |
| `L1H12_mean_component` | `SUPPORTED` | Mean-ablation partner / pseudo / specific effects are `+0.2407` / `-0.0040` / `+0.2447`; bounded component pattern (`AC5`). |
| `L4H11_noising_component` | `SUPPORTED` | Noising partner / pseudo / specific effects are `+0.0952` / `+0.0018` / `+0.0934`; bounded component pattern (`AC6`). |
| `L4H9_noising_component` | `SUPPORTED` | Noising partner / pseudo / specific effects are `+0.0917` / `+0.0011` / `+0.0905`; bounded component pattern (`AC6`). |
| `L4H7_noising_component` | `SUPPORTED` | Noising partner / pseudo / specific effects are `+0.0790` / `+0.0002` / `+0.0788`; bounded component pattern (`AC6`). |
| `distinct_acceptor_route_candidate` | `INCONCLUSIVE_METHOD_DISAGREEMENT` | Holm-adjusted `p=0.00200` passes, but the specific delivery-gap interval begins at `+0.0482`, below the `+0.05` boundary. No distinct-route claim. |

This is a clear but limited result: R3c supports only the `L1H12` mean-ablation and
`L4H11`/`L4H9`/`L4H7` same-example-noising component patterns. It does not support `L3H1` reuse, a
distinct route, or `L2H15`; it establishes no address, payload, acceptor-side write, delivery,
mediation, sufficiency, faithfulness, complete circuit, or biological mechanism. Same-example noising
is not delivery.

The existing R5 chain therefore stops. Its fail-closed runners have model-free audit evidence only,
their literal pins remain unset, and R5a correctly rejects this R3c entry because supported `L3H1`
reuse is mandatory. Supported secondary heads cannot be substituted after seeing the target. The next
licensed design choice is either (a) a prospectively frozen, gene-disjoint clarification of the
registered `L3H1` path, followed by the existing R5 chain only if it passes, or (b) a separately frozen
acceptor-route mechanism program centered on `L4H11`/`L4H9`/`L4H7`. Option (b) must add restoration and
mediation tests before using delivery language.

## 2026-08-17 — the canonical causal adjudication closes at the frozen R1c→R5 handoff

The choice above was resolved prospectively, not by substituting one of R3c's supported secondary
heads. A separate R3d adjudication ran on a complete registry-disjoint train target and supported its
registered `L3H1_reused_component` Path A. The exact target artifact and manifest have SHA-256
`e33d37447cbba368a48beea108fc49c06c56adfbeef2c0f95f019c227d60cb88` and
`53f13aa4884b53a91eba6efa2091c7e7638672ed260bf6558f20547d88f899cd`. Its distinct-route Path B was
threshold-close and distribution-sensitive, so the final R1c→R5 design admitted **Path A only** as a
component prior. `L2H15`, the optional low-rank state, H7, and all GC/GT or early-head hypotheses were
fixed as secondary diagnostics and could not gate, rescue, or redesign the chain.

The entire behavior→reader→post-`W_O` local-write→L4H11+H9-delivery stack was then frozen before a new
outcome. The R1c validation preflight ran once from clean commit `9008ba3` with strict MPS execution and
CPU fallback absent. It used 400 outcome-blind validation rows / 321 genes on chr11/16/19, materialized
no chr8/9 row in that run, and scored none of the R1c target outcomes (`ACR18`). The result was a genuine `PREFLIGHT_GO`: all 13 registered gates passed. Three
independent model-free checks authenticated or replayed the package, including exact row/cell/family
algebra, the 4,000-draw sign tests and simultaneous intervals, both 10,000-draw whole-gene robustness
families, and all 1,000 calibration trials. The controlling identities are:

- artifact `1b27c4132c71a576c435b7b71f32831af9a44fbca6325931f35716ae4133cbdd`
  (`74,352,692` bytes);
- manifest `7412f3159e82f6ab0bbaa95e2796f41aa038eef4233efa7f09ec65867e39c45c`
  (`11,370` bytes);
- public-verifier attestation
  `04be60c2239c98eefb01521f7a8a94a68f4d2cc4db5db8f1dddb5622f7378ce1`.

That passing preflight did **not** authorise the sealed target. Before installing any consumer or target
pin, the exact frozen cross-stage loader was exercised on the reviewed package. R1c `/3` correctly
serializes six margin fields per row (`P_clean`, `P_under_D`, `P_under_N`, `X_clean`, `X_under_D`,
`X_under_N`) and its own identity projects the three `P_*` common legs. The frozen R5 helper instead
requires `margin_levels` to contain exactly those three keys before projection, so it rejects the
authentic package with `R5CapstoneCohortError: R1c partner margin levels differ`.

The technical change is obvious only **after** the validation result: accept the authentic six-field
record and project the three common legs. That helper is part of the frozen joint stack
`b89efd7c0d83d579aa7e0a2d36f8059c96913242a1c943ec64d87fe7590b00bc`. Patching it now creates a new
method after an outcome; rewriting the artifact or rerunning the preflight violates the same one-run
rule. The decision is therefore to make no repair, install no package pin, and stop. `ACR19` records the
defect and `AC9` is the study's single terminal outcome: **`INSTRUMENT_UNRESOLVED` before sealed target**.

Consequences are exact: the R1c target seal remains `None`; all shared R1c package pins remain `None`;
no R1c target attempt or completion exists; no R5a, R5b, or R5c scored artifact exists; and this
program never scored the R1c masked-acceptor target outcomes on chr8/9. The strong validation-preflight estimates are instrument evidence only. They are not a
behavior result and cannot be promoted into evidence for L3H1 reading, a post-`W_O` write, conditional
L4H11+H9 delivery, mediation, sufficiency, or a biological mechanism.

## 2026-08-17 — consolidate the natural donor-class modifier without promoting it into the stopped causal chain

The independent donor-class history at merge parent `b79570d` is retained in full, but its scientific
record is reconciled rather than overlaid on AC9. The exact first M6 run—not the prohibited second
run—is restored as the live compact package. It was produced cleanly at `42954a6` under the original
protocol hash `da326e4d…f46a`; the artifact and manifest hashes are `80150013…051a` and
`4e470908…16df`. The later source-branch package remains reachable through the merge parent as
superseded provenance. `ACR21` records why the lifecycle, checkpoint, cohort-identity, bootstrap, and
replay surface bound the evidence despite the numerical replication.

The ledger splits M6 by evidence rung. AC10 records the masked-acceptor behavior modifier
(`rho=0.2571`, upper bound `0.5402` against `0.60`); AC11 records L3H1 donor-attention geometry
(`0.3755`, upper bound `0.5809` against `0.70`). Both are bounded held-out verification on 67 matched
pairs / 56 treatment genes. Neither is called confirmation, component necessity, mediation, or a
mechanism. M1–M5 remain adaptive development: their reusable matching/estimator code and runners are
kept, while scratch-only outcomes are not promoted into the results manifest (`ACR22`).

Two compatibility decisions keep the consolidated tree reviewable without invalidating the frozen
R1c/R5 package. First, historical `circuit.eligible` remains byte-identical and GT-only; the natural
modifier runners opt into `GC` through `donor_class.eligible_donor_motifs` (`ACR20`). Second, the
duplicate MLP hook is not retained: M5 uses the already-reviewed `edit_mlp_writes` primitive. The exact
historical M6 runner and source state remain recoverable at `42954a6`; current development runners are
adapted only outside the frozen core.

The scientific storyline is additive, not synthetic. AC1–AC8 delimit canonical `GT-AG` behavior and
component candidates; AC9 closes the prospective read→post-`W_O` write→delivery adjudication before
target; AC10–AC11 show that natural `GC-AG` donor class modulates behavior and attention. No combination
of these rows establishes causal construction of the masked-acceptor margin, so no new scored target or
follow-up is launched by this consolidation.

## 2026-08-17 — retain M4d as a score-margin correction, not a new claim

The later Claude worktree added a useful log-odds/QK diagnostic after the natural donor-class branch
had already been merged. It had no new branch commit: the full M4d execution ran once from a dirty
worktree above `b79570d`. Post-run snapshot `773c9e8987e05e2605974004795f92f67425cd07`
preserves the recovered runner, score arithmetic, capture hook, and tests, but is explicitly not the
execution commit. The exact development artifact and manifest are retained under
`results/splicebert_acceptor_coupling/development/` at SHA-256 `d1d96029…f763` and
`879b262b…f165`.

The score-margin identity is methodologically useful, and its pointwise results narrow M4c's broad
labels to three nominal layer-0 candidates (`L0H14`, `L0H5`, `L0H3`), three unresolved earlier labels
(`L0H4`, `L1H6`, `L1H13`), and one opposite-direction `L2H15` diagnostic. They are not promoted. M4d
changed the batch-dependent mean intervention from M4c's batch 2 to batch 4, clustered only treatment
genes, omitted family correction and matched-size nulls, used an ordered reference-arm decomposition,
and retained neither rows nor draws. Its score-row shifts cannot separate query from key effects. The
artifact's `1.5567x` “ceiling” is corrected to a collapsed-baseline toy benchmark (`ACR23`).

The consolidation therefore adds no AC12 and changes no AC1–AC11 disposition. It also leaves the
byte-bound R1c/R5 `hf_bert.py` untouched: the reviewed score capture lives in a new module, while the
adapted development runner uses the consolidated donor-class eligibility helper, connected two-arm
gene-component resampling, and strict JSON.
No model, genome, cohort, or reserve is rerun for this integration.
