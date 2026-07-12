# The sealed test — chr8 / chr9, opened once

**13/13 pre-registered checks PASS.**

Run against the protocol frozen in `PREREGISTRATION_SEALED.md` (hashes recorded there before
the split was touched). No component was selected on test data; no threshold was moved after
seeing it. `results/splicebert_coupling/sealed_test.json`.

---

## Replication

| | val (chr11/16/19) | **test (chr8/9) — SEALED** | gate |
|---|---|---|---|
| **BEHAVIOUR** | | | |
| unconditional coupling | +0.958 [0.855, 1.059] | **+0.872 [0.771, 0.969]** | A1 ✅ |
| median (the effect is right-skewed) | 0.714 | 0.638 | |
| vs distance-matched other donor | +0.031 → diff +0.927 | **+0.005 → diff +0.867 [0.757, 0.971]** | A2 ✅ |
| vs non-partner AG | +0.012 → diff +0.945 | **+0.007 → diff +0.865 [0.757, 0.970]** | A3 ✅ |
| partner selectivity | 15.6× [8.4, 39.9] | **19.6× [12.0, 37.2]** | |
| **NECESSITY** | | | |
| ablate L3H1 | −82.3% | **−78.7%** | |
| ablate L2H15 | −52.2% | **−47.7%** | |
| ablate C = {L3H1, L2H15} | −86.8% | **−81.6%** | B1 ✅ |
| MLM health retained (masked probes) | 97.8% | **99.2%** | B3 ✅ |
| matched-random null (400 distinct 2-head sets) | p = 0.0025 | **p = 0.0025** | B2 ✅ |
| **MEDIATION — the conjunctive read** | | | |
| attention collapse under corruption | 94% | **93%** | |
| key only | 0.031 | **0.032** | |
| value only | 0.117 | **0.101** | |
| **key + value** | **0.680** | **0.663** | C2 ✅ |
| **K×V interaction I** | **+0.532 [0.478, 0.583]** | **+0.531 [0.477, 0.583]** | C1 ✅ |
| key+value @ control position | −0.000 | **−0.000** | |
| **BINDING — the relocated read** | | | |
| value alone @ decoy *(must be ~0)* | 0.004 | **−0.010** | D2 ✅ |
| **CONTROL:** key @ non-site *(must be ~0)* | 0.100 | **0.073** | D3 ✅ |
| key alone @ decoy AG | 0.238 | **0.244** | |
| **key + value @ decoy AG** | **0.669** | **0.655** | D1 ✅ |
| key + value @ a non-site | 0.656 | **0.639** | |
| key + value in place (reference) | 0.686 | **0.658** | |
| relocation retains | 98% | **100%** | D4 ✅ |
| **FAITHFULNESS** | | | |
| patch C restores | 83% | **79% [74%, 83%]** | E1 ✅ |

The K×V interaction — the single number that says the read is *conjunctive* rather than
additive — reproduces to three decimals on chromosomes the circuit has never seen:
**+0.532 → +0.531**.

## What the sealed test establishes

1. **The behaviour is real and partner-specific.** Destroying the partner's `AG` costs the
   donor 0.872 of logit margin. Destroying a *motif-matched non-partner* `AG` costs 0.007.
   The same corruption, scored at a *distance-matched other donor*, costs 0.005.
2. **C = {L3H1, L2H15} is necessary and specific.** Ablating it removes 82% of the coupling
   while leaving 99.2% of the model's masked-nucleotide accuracy intact. No random 2-head set
   comes close (p = 0.0025 against 400 distinct matched sets).
3. **The read is conjunctive.** The acceptor's key selects it (restores 79% of attention but
   only 0.032 of the coupling); its value supplies the content (restores 0% of attention and
   0.101 of the coupling); together they restore 0.663. I = +0.531.
4. **The address is not a coordinate.** Copy the partner's (key, value) pair to a decoy AG and
   the coupling returns (0.655); copy it to a position that is not a splice site at all and it
   still returns (0.639); copy it back where it came from and you get 0.658. **Relocation costs
   nothing.** Copy only the content, with no address, and you get −0.010.

## What it does not establish

- **Completeness.** The obvious test has no power — see `CLAIMS.md` D15. C is the *dominant
  sub-circuit*, not "the circuit". **L1H12 (0.266 on val) is real, outside C, and unexplained.**
- **Long introns.** The donor→acceptor gap must fit the 1024 nt context (60–150 nt here).
  Typical human introns are orders of magnitude longer. This architecture cannot test them.
- **Any other model.**

## Provenance

- Frozen protocol: `PREREGISTRATION_SEALED.md` (code hashes recorded pre-unblinding).
- Validation before the seal: `--split val` → 13/13.
- The split was opened exactly once, by
  `run_sealed_test.py --split test --i-understand-this-unseals-the-test-split`.
- Four blockers and three retracted published numbers were fixed *before* the seal, not after.
  Read `CLAIMS.md` §Retractions (R7–R12) before quoting any pre-2026-07-12 figure.
