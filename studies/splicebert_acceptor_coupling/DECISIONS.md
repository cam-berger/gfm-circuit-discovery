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
