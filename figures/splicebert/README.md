# SpliceBERT phase figures — the state of the record

**Everything here is GENERATED** from `results/phase-lattice/fig_phase_data.json` (+ the run
artifacts beside it), so no figure can drift from the result it claims to show. Each has a
`.CAPTION.md`; **the narrative lives there, not in the artwork** — the figures are held to a few
words per component.

| figure | script | what it shows |
|---|---|---|
| `fig_phase_circuit` | `make_fig_phase_circuit.py` | the mechanism: clock → L2H15's address → L3H1's read → coupling |
| `fig_phase_clock` | `make_fig_phase_clock.py` | the evidence: P's spectrum, the wave in six heads, collapse to chance, dose-response, provenance |
| `fig_phase_wave` | `make_fig_phase_wave.py` | the readout itself — the comb **is** a cosine, and **L0H11 needs two of them** |
| `fig_two_carriers` | `make_fig_two_carriers.py` | the second carrier: P holds **two** clocks, and they delete independently |

---

## The one-paragraph version

SpliceBERT's learned position table `P` [1026 × 512] is **a small bank of oscillators**. Two of
them are fast: a **period-3 codon clock** (bins 339–345) and a **second carrier at ~10.67**
(bin 96). Both are near-perfect **circles** — rank-2 rotary codes — and therefore **have no phase
of their own**; each attention head *chooses* a phase by choosing a direction in the carrier's
plane. Six heads read the codon clock as a comb; **one of them, L0H11, reads both carriers**, and
its complete wave function needs two cosines. The codon clock carries the splicing circuit. The
second carrier does not, and **what it is for remains open.**

## The claims, and their exact status

| claim | status |
|---|---|
| Transformers learn oscillator banks | **established** (RoBERTa, trained on *text*, has one) |
| SpliceBERT selectively amplifies a **sharp** ~10.67 tone | **established** (76% of its 9–12 band in one bin, vs RoBERTa's broadband 20%) |
| L0H11 uses that tone | **established** (D23 — and post-softmax, not just in the logits) |
| Genomic/chromatin statistics **selected** the tone | **plausible, UNTESTED** |
| The model uses it to represent chromatin | **unproven** |
| The 512-position grid **quantized** a broad nucleosome-scale band into bin 48 | **new, concrete, testable** (D24) |

⚠️ The selection is **model/data-specific**. It could still be driven by a *non-biological*
property of training — fixed length, optimisation, corpus size, repeats. **"Biological" is not
written until the bridge is crossed.**

## Two things that are NOT evidence, and must never be quoted as such

1. **R16 — the chromatin reading is OPEN, not refuted.** An earlier null ("the sequence's WW/SS
   helical phasing does not modulate the damage") came from an estimator whose **detection floor
   is 5–10% modulation** against a literature effect of *a few percent*. The instrument was
   validated only *after* it gave the expected answer. It fails.
2. **Assay 1's p-values.** The unanchored corpus spectrum found nothing at 10.6667 (p ≥ 0.39),
   but it is a **quadratic** detector (SNR ∝ depth²·√N). Its only reportable result is its
   calibrated sensitivity limit: *at 120 Mb it can exclude only large, globally incoherent
   effects — not the expected few-percent nucleosomal signal.*

The experiment that **can** see it is the **dyad-anchored** assay (D25): a linear detector
(SNR ∝ depth·√N), built and power-calibrated — 80% power at 2% needs only ~3,000 well-positioned
dyads. The binding constraint is **dyad precision, not dyad count** (2 bp of jitter costs the same
as discarding four fifths of the dyads), which is why it needs a **chemical-cleavage** map, not
MNase-seq.

## Superseded

`splicebert_donor_acceptor_binding.svg` — asserts the **retracted** "content-addressed read"
(R15). Replaced by `fig_phase_circuit.svg`. See `HAND_ASSETS.md`.
