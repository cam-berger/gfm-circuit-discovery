# DNABERT-2 core promoter: TATA→TSS study

> **Tier: T2 — provenance.** Study record. The claim ledger is the source of truth for wording.

Second worked example of the repository method, on a model chosen for **architectural contrast** with
SpliceBERT rather than for similarity (`docs/TRANSFER_GUIDE.md`).

## Study identity

| | |
|---|---|
| **Model/checkpoint** | `zhihan1996/DNABERT-2-117M`, revision `7bce263b15377fc15361f52cfab88f8b586abda0`, frozen |
| **Tokenizer / position scheme** | BPE, vocab 4096, **~4.66 nt/token**; **ALiBi** — *no learned position table* |
| **Objective / corpus** | masked LM; multi-species (human included) |
| **Biological computation** | use of TATA identity **and** native TATA-to-TSS geometry in reconstructing a masked TSS-centred span |
| **Model-native metric** | length-normalised true-token **pseudo**-log-likelihood at the simultaneously masked target tokens |
| **Primary dataset** | EPDnew human Rel. 006 (29,598 promoters) × hg38 |
| **Compute budget** | CPU/MPS; ~20 short forward passes per promoter |

## Contrast with SpliceBERT — why this model

| Axis | SpliceBERT | DNABERT-2 |
|---|---|---|
| Tokenization | single-nucleotide | BPE, variable-length, content-dependent |
| Position | absolute learned table (P=1026) | ALiBi attention bias, **no table** |
| Norm | post-LN (exact reconstruct-inject) | MosaicBERT, internal unpadding |

Two consequences are load-bearing and recorded in
`results/origin-crossmodel/canonical/dnabert2_aliasing.json`:

1. **The period-3 carrier family cannot be ported.** A 3-nt period is 0.64 token-steps — below the
   Nyquist period of 2. It is not representable in token index. A null here would be a tokenizer
   artifact, not a result.
2. **Per-nucleotide internal-state claims are invalid.** All interventions are designed in nucleotide
   coordinates and executed over mapped token spans.

## Documents

| File | Role |
|---|---|
| `G0_GATE_PACKET.md` | Phase-0 computation lock: frozen question, evidence table, rivals R0–R9, gate decision request |
| `preregistrations/PREREGISTRATION_TATA_TSS.md` | the protocol — estimands, memorials, calibration, decision rule, and the rejected-alternatives record from two design red-team rounds |
| `DECISIONS.md` | dated decisions and deviations |
| `CLAIMS.md` | claim ledger — **empty by design**; nothing is claimed yet |
| `requirements-dnabert2.txt` | pinned execution environment, and why it is study-local |

## Gate status

| Gate | State |
|---|---|
| **G0** computation lock | **evidence packaged, awaiting gate-owner decision** |
| **G1** cohort + counterfactual integrity | not run — two named blockers (see G0 packet §6) |
| **G2** simple explanations bounded | not run |
| **G3** native MLM behaviour | not run; **confirmation split sealed** |
| Phases 4–7 | deliberately unlicensed |

**No agent may advance a gate or promote a claim.** Agents supply evidence; the gate owner decides.

## Running

Canonical invocation from repo root (the study is pinned to `transformers==4.44.2`; see
`requirements-dnabert2.txt` for why it is not a root extra):

```bash
uv run --isolated --no-project --python 3.11 \
  --with-requirements studies/dnabert2_core_promoter/requirements-dnabert2.txt \
  --with-editable . \
  python scripts/dnabert2/<runner>.py [--smoke]
```

Every runner asserts the pin, re-hashes the vendored code and the checkpoint, and embeds the result in
its manifest. A version or hash mismatch is a hard stop.

## Assets acquired (this branch)

| Asset | Location | Provenance |
|---|---|---|
| DNABERT-2 modelling code | `src/gfmcd/adapters/dnabert2/vendor/` | verbatim @ pinned revision, sha256 in `VENDOR.sha256` |
| Checkpoint | local HF cache (never downloaded at run time) | sha256 `7ff39ec7…7aeba`; MLM head complete |
| EPDnew human Rel. 006 | `data/external/epdnew/Hs_EPDnew.{dat,bed}` | sha256 in `EPDNEW.sha256` |
| TBP PWM | `data/external/epdnew/JASPAR_MA0108.3_TBP.json` | JASPAR MA0108.3, 7 nt, consensus `TATAAAA` |
| hg38 | `~/Research/data/genomes/hg38.fa` | UCSC goldenPath, md5 `b2aee9f8…ee63` |
