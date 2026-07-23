# Cross-model carrier transfer (origin-crossmodel)

> **Tier: T1 — canonical.** Phase 2 of the positional-carrier origin study.

**Question.** Are SpliceBERT's two learned positional carriers — the **period-3** (codon-scale)
carrier and the distinct **≈10.63** carrier — idiosyncratic to SpliceBERT, or do other
single-nucleotide transformer MLMs build them too? This is a natural experiment on frozen
checkpoints: no retraining. The key contrast is training domain.

- **SpliceBERT** — trained on spliced pre-mRNA (**ORF-rich**). Both carriers present (Phase 1).
- **RNA-FM**, **RNA-BERT** — trained on **ncRNA (largely ORF-free)**. Same tokenizer family
  (MultiMolecule `RnaTokenizer`, single-nt, vocab 28), same two defining objects (a learned
  absolute position table P; attention heads). This is where the carriers should be **absent**
  if period-3 is a **codon artifact** built from reading frames in the training corpus.
- **DNABERT-2** — BPE-4k tokenizer. A deliberate **tokenizer-aliasing contrast**, not a
  convergence test.

**Controlled nomenclature (enforced).** "≈10.63 carrier" (period 1026/96 = 10.6875 nt). The DNA
nucleosomal signal ≈10.298 nt is a **distinct** quantity and never conflated. period-3 ≠ ≈10.63.

## Headline result

Neither carrier exists in either ncRNA-trained model. This is a **powered null**, not
insensitivity: injection calibration shows a carrier of depth ≥5% (RNA-FM) / ≥20% (RNA-BERT) of
the P scale would be detected at 100%, with a depth-0 false-positive rate of 0.

| Model (domain) | period-3 (P) | ≈10.63 (P) | Attention | Behavior | Verdict |
|---|:--:|:--:|:--:|:--:|---|
| SpliceBERT (spliced, ORF-rich) | ✓ Z=241 | ✓ Z=456 | ✓ | ✓ | carriers present (Phase 1) |
| **RNA-FM** (ncRNA) | **null Z=0.95, p=0.80** | **null Z=1.05, p=0.25** | null | n/a | **both carriers absent** |
| **RNA-BERT** (ncRNA) | **null Z=1.05, p=0.45** | **null Z=1.31, p=0.26** | null | n/a | **both carriers absent** |
| DNABERT-2 (BPE-4k) | no learned P | no learned P | — | — | tokenizer cannot express it |

Z = leakage-aware carrier band power / median control-period band power (Z≈1 ⇒ no carrier).
The behavioural (donor–acceptor coupling) column **does not transfer**: ncRNA has no introns /
splice sites, so structural presence in P + read into attention + generic MLM use is the ceiling.

**Interpretation (origin).** The codon-origin prediction is confirmed on the P column: the
period-3 carrier is present in the ORF-rich SpliceBERT and **absent** in two ORF-free ncRNA
models measured with the same, injection-calibrated instrument. Combined with Phase 1's
human-only sibling contrast, this is consistent with period-3 being **built by the reading-frame
statistics of the training corpus** rather than being a generic property of single-nt RNA MLMs.
The ≈10.63 carrier is likewise absent in the ncRNA models. Both are convergence-vs-idiosyncrasy
datapoints; absence is a positive, reportable result here.

## Files (canonical/)

| Artifact | Assay | Evidence column | What it shows |
|---|---|:--:|---|
| `acquisition_validation.json` | acquisition | — | RNA-FM/RNA-BERT load through the existing `SpliceBertAdapter` (`tokenizer_mode='raw'`); single-nt confirmed; absolute learned P; config + weight SHA-256 recorded. |
| `carrier_pspectrum.json` | P-spectrum detection | **P** | Leakage-aware + exact-bin band power at period-3 and ≈10.63 for both models, each with a phase-scramble null, control-period rank test, injection detection curve, and depth-0 FPR. **The headline null.** |
| `reader_and_surgery.json` | reader scan + control surgery | **Attention** | All-heads post-softmax coherence `m_T` (bounded in [0,1]) at both periods, per-head permutation p, BH-FDR; would-be-band deletion vs matched control bands. |
| `reader_freqcontrol.json` | reader frequency control | **Attention** | Refutes the apparent "16 RNA-FM heads at ≈10.63": the significant-head count at 10.6875 is **not** a peak above a dense control-period grid (14 vs grid max 28) — an envelope/permutation-null artifact. Continuous logit-shift surgery channel (top-1 accuracy is degenerate on random OOD sequence for these models). |
| `dnabert2_aliasing.json` | tokenizer aliasing | methodological | DNABERT-2 has **no learned position table** (ALiBi backbone; `position_embeddings_keys=[]`). Even if it did, BPE tokens span mean 4.66 nt (1–10, σ=1.36); a codon (3 nt) = 0.64 token-steps, below the token-index Nyquist of 2 ⇒ unrepresentable. A null here is a **tokenizer artifact**, not evidence of absence. |

Each `*.json` has a sibling `*.manifest.json` (git rev, environment, checkpoint/config/weight
SHA-256, tokenizer, params, seed, command). **Note on P weight hashes:** `acquisition_validation`
records the native **float32** weight SHA-256 (e.g. RNA-FM P = `2702e198…`); the spectral assays
cast P to **float64** for numerical work and record that byte encoding (`f135d23c…`). Both are the
same tensor, verified reproducible; the encodings differ, not the weights. Instruments:
`src/gfmcd/crossmodel/`
(`carrier_spectrum.py`, `reader_scan.py`); runners + tests: `scripts/crossmodel/`
(`test_carrier_spectrum.py`, 7/7 pass). Figure: `figures/origin-crossmodel/`.

## Claim scope (evidence firewall)

- **P column — canonical.** period-3 and ≈10.63 are absent from RNA-FM and RNA-BERT position
  tables (powered null). This is the claim the data reaches.
- **Attention column — supporting.** The absence propagates into post-softmax attention: no head
  reads either carrier above the frequency-control baseline.
- **Behavior column — not claimed.** The splicing coupling arm does not transfer to ncRNA.
- **DNABERT-2 — methodological.** Demonstrates why single-nt tokenization is necessary to *express*
  a codon-scale positional code; says nothing about whether the genome carries one.
