# Claim ledger — DNABERT-2 core promoter

> **Tier: T1 — canonical** for this study's claim wording.

## Live claims

**None.** No confirmatory experiment has been run. This is deliberate and is the correct state at G0.

## Established instrument facts (not scientific claims)

These are properties of the checkpoint and the data, established by preflight. They license no statement
about model behaviour or biology.

| # | Fact | Evidence |
|---|---|---|
| **F1** | DNABERT-2-117M carries a **complete, trained MLM head** (`cls.predictions.decoder [4096, 768]`; `missing=[]`, `unexpected=[]`) | direct state-dict load, rev `7bce263b…abda0` |
| **F2** | The checkpoint has **no learned position table** (ALiBi). `config.position_embedding_type: "absolute"` is misleading | no `position_embeddings` key in 142-key state dict |
| **F3** | The EPD→hg38 coordinate convention reproduces the EPD `SE` line **3,000/3,000**, transcription-oriented | memorial M1 |
| **F4** | The TATA motif-start distribution peaks at **−30/−31** with 89% of mass in −33…−29, under both a strict consensus and JASPAR MA0108.3 | memorial M7 |
| **F5** | Under MA0108.3 (rel ≥ 0.85, start in −33…−27) the EPDnew human cohort is **3,297 / 29,598 (11.1%)** | cohort scan |
| **F6** | BPE retokenisation from an 8-nt edit reaches median 2 nt, p90 7, max 16 downstream; target-token invariance holds for 99.8–100% of pairs at a ≥20-nt buffer | tokenizer probe, 400 real promoters |
| **F7** | Naive TATA-vs-filler arms differ in **total token count (31–39%)** and **inter-slot token boundaries (45–49%)** — a mechanical edit→target channel under ALiBi | tokenizer probe, 400 real promoters |
| **F8** | That imbalance is **removable by construction**: a composition- and token-geometry-matched filler exists for **98.3–100%** of promoters (median 32–43 candidates) | constrained-filler search |

**F7/F8 are the load-bearing pair.** F7 says the obvious design is confounded in the majority of cases;
F8 says the confound can be designed out rather than modelled away. Together they justify memorial M2b.

## Retractions / corrections

None yet.

## Next bridge

Gate-owner decision on **G0** (and blocker **B1**), then Phase-1 cohort construction. The first
scientific question that can be answered is whether `ell` responds to **anything** 20+ nt from the
target — measured by the graded copy-plant. A null there stops the study at preflight with a diagnosed
instrument limit, which is a valid result.
