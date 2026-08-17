# Decision log — SpliceBERT acceptor coupling

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

**Two prior results are folded in rather than re-derived**, and the observation that opened this study
is that they agree on something neither states. `AR1`'s `DONA` arm measured ρ ≈ 0.62 for every donor alt
with `z[1] == "T"`; `E11` measured `Retention_A` of 3% at two substitutions against 22% at one. Those
are the same dose-response — **a donor destroyed with one substitution leaves the `T` at `d+1` standing
and retains most of its effect** — reached from different cohorts by different assays, on branches that
do not cite each other. R1 treats edit size as a factor to be measured rather than a nuisance to be
controlled away.

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

**The genuinely new result is `L2H5`.** It sits in the same layer as the forward addresser, is
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

## Held-out component confirmation returns a mixed bounded outcome (legacy run `R3c v2.1`; 2026-08-15)

The successor to AC4 was narrowed to one masked-acceptor endpoint, ten heads, two intervention modes,
a 62-row simultaneous-inference family, and seven named decision paths. It excludes historical R3 genes
and divides the remaining validation cohort into a preflight bucket and a gene-disjoint target bucket.

Two authorization defects were closed before target science. The first rule could have revived an
`L2H15` preflight-health failure using target health; authorization is now monotone (`ACR16`). The next
runner inspected a supplied model path before rejecting a missing preflight artifact; the repaired guard
runs before any caller-supplied target input, and the changed runner required a fresh preflight (`ACR17`).
Neither predecessor opened bucket-1 outcomes.

The clean successor preflight authorized the 500-row / 304-gene bucket-1 target on validation
chromosomes 11, 16, and 19. The active
[attestation](../../results/splicebert_acceptor_coupling/attestations/paired_donor_component_confirmation_v1.json)
pins the exact archived package, byte sizes, SHA-256 values, and clean execution commits. The
[decision audit](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json)
reports `AUDIT_PASS_MIXED_SCIENTIFIC_OUTCOME` after two exact replays without rescoring the model or
reopening the cohort.

| Reader-facing evidence | Legacy alias | Frozen disposition |
|---|---|---|
| `acceptor-coupling/components/l1h12-mean-ablation` | `AC5` | `SUPPORTED` |
| `acceptor-coupling/components/l4-head-noising` | `AC6` | `SUPPORTED` for `L4H11`, `L4H9`, and `L4H7` |
| `acceptor-coupling/components/route-hypotheses-inconclusive` | `AC7` | `INCONCLUSIVE_METHOD_DISAGREEMENT` for reused `L3H1` and the distinct-route conjunction |
| `acceptor-coupling/components/l2h15-health-ineligible` | `AC8` | `HEALTH_FAILED` under the monotone authorization ceiling |

This result supports only bounded, intervention-specific component patterns. Same-example noising is not
delivery. The failed registered conjunctions cannot be replaced post-outcome by supported secondary
paths, and the record establishes no address, QK role, post-`W_O` write, directed edge, delivery,
mediation, sufficiency, faithful circuit, or biological mechanism.

## 2026-08-17 — close the frozen mechanism handoff instead of repairing it after outcome

The successor program froze behavior validation and the downstream reader → post-`W_O` write → delivery
consumer as one stack. Its clean 400-row / 321-gene validation preflight passed all 13 gates, and the
archived audits reproduced its statistical and calibration decisions. Before any target or downstream
stage was opened, however, the authentic package failed the exact frozen consumer: producer rows contain
six `P_*`/`X_*` margin fields while the consumer demands exactly three `P_*` fields before projecting
those same fields.

The obvious parser repair became obvious only after a successful validation outcome. It would change the
frozen stack; rewriting the package or rerunning would violate the same one-run rule. The decision is to
make no repair, install no target authorization, and stop. The semantic record
`acceptor-coupling/instrument/frozen-handoff-schema-mismatch` (legacy `AC9`) is therefore terminal
`INSTRUMENT_UNRESOLVED` before target science. No masked-acceptor target outcome on chr8/9 and no
downstream reader, write, or delivery result exists (`ACR19`).

## 2026-08-17 — keep natural donor-class modifiers orthogonal to the stopped mechanism program

The natural donor-class reserve addresses a different question: whether `GC-AG` donor class modifies
masked-acceptor behavior and `L3H1` attention relative to matched `GT-AG` donors. The clean first run on
67 pairs / 56 treatment genes numerically crossed both preregistered boundaries: behavior ratio `0.2571`
with upper bound `0.5402 < 0.60`, and attention ratio `0.3755` with upper bound `0.5809 < 0.70`.

The decision is to retain that first clean package and the original protocol, while excluding the later
prohibited rerun from the active record. Checkpoint, cohort-identity, bootstrap, and replay limitations
bound both results to verification (`ACR21`). They become two records at their actual evidence rungs:
`acceptor-coupling/natural-gc/behavior-modifier` (legacy `AC10`) and
`acceptor-coupling/natural-gc/l3h1-attention-modifier` (legacy `AC11`). They do not repair AC9, show
component necessity, or complete a route. Canonical `GT-AG` scope and the quasi-observational natural-
`GC` comparison remain explicit and separate (`ACR20`).

## 2026-08-17 — retain the QK score measurement as a development diagnostic, not a claim

The archived pre-softmax donor-versus-other score diagnostic is useful because it replaces broad
attention-probability shorthand with the exact score margin. It nominally prioritizes `L0H14`, `L0H5`,
and `L0H3`; leaves `L0H4`, `L1H6`, and `L1H13` unresolved on that scale; and gives `L2H15` an
opposite-direction diagnostic.

It does not enter the claim ledger. The candidates were adaptive, their intervals are
uncorrected and lack matched-size nulls, clustering omits comparator-side dependency, and the historical
package is dirty and incomplete. More importantly, the `GC` and `GT` arms were scored under separately
recomputed batch×sequence-conditioned mean replacements, so their difference is not intrinsic class
specificity. The score change also cannot distinguish acceptor-query from donor- or competitor-key
effects (`ACR22`, `ACR23`). The semantic ID
`acceptor-coupling/natural-gc/qk-score-diagnostic` therefore names a development-only record. No model,
genome, cohort, or reserve is rerun for this consolidation.
