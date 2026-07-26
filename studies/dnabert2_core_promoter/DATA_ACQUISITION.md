# Data acquisition and provenance

> **Tier: T2 — provenance.** `data/external/` is gitignored, so this file *is* the tracked record.
> Re-fetch with `scripts/dnabert2/fetch_data.sh`, which verifies every hash below.

Acquired 2026-07-26.

## EPDnew — human promoters

| | |
|---|---|
| Source | `https://epd.expasy.org/ftp/epdnew/H_sapiens/current/` |
| Release | **Rel. 006** (`DT` lines), 29,598 promoters |
| Assembly | **hg38**, UCSC chromosome naming (matches the genome FASTA below — no lift-over, no chr-name mapping) |

| File | sha256 | Local path |
|---|---|---|
| `Hs_EPDnew.dat` | `88cfaf67ce2ad7a84ac5be779a54e9ded443e626137936526e37a839a06f0447` | `data/external/epdnew/` |
| `Hs_EPDnew.bed` | `3ecac81b26ed99cc4d91e84f9dedd123de250a42981920d8d065e09c2ca114f8` | `data/external/epdnew/` |

**What each file supplies.** `.bed` gives `chrom, start, end, name, score, strand, thickStart, thickEnd`
— the 60-nt EPD window and the TSS anchor. `.dat` gives the gene symbol, cross-references, and the `SE`
line: the same 60 nt as **49 lowercase (−49…−1) + 11 uppercase (+1…+11)**, which is what makes the
independent coordinate check in memorial **M1** possible.

**Not supplied by either file:** any TSS shape/focus score, and any TATA annotation. The first is
blocker **B1**; the second is *deliberate* — scoring TATA ourselves with a frozen PWM means the study
controls the coordinate anchor outright, which is how rival **R6** is closed rather than inherited.

## TBP position-frequency matrix

| | |
|---|---|
| Source | `https://jaspar.elixir.no/api/v1/matrix/MA0108.3/?format=json` |
| Matrix | **MA0108.3**, TBP, CORE vertebrates, **7 positions**, consensus `TATAAAA` |
| Local path | `data/external/epdnew/JASPAR_MA0108.3_TBP.json` |
| Frozen use | log-odds vs uniform background, pseudocount 0.25; **relative threshold 0.85**; **motif-START** anchor; functional interval **−33 … −27** |

Derived constants (recomputed and asserted at run time, not hard-coded):
`max = 11.85`, `min = −44.19`, `threshold = 3.448`, **95 / 16,384** 7-mers pass.

## Reference genome

| | |
|---|---|
| Source | UCSC goldenPath `hg38.fa.gz` |
| Local path | `~/Research/data/genomes/hg38.fa` (+ `.fai`) |
| md5 | `b2aee9f885accc00531e59c4736bee63` |

Outside the repository and shared with other studies; referenced by path, never copied.

## Model

| | |
|---|---|
| Checkpoint | `zhihan1996/DNABERT-2-117M`, revision `7bce263b15377fc15361f52cfab88f8b586abda0` |
| Location | local HF cache — **never downloaded at run time** |
| `pytorch_model.bin` | sha256 `7ff39ec77a484dd01070a41bfd6e95cdd7247bec80fe357ab43a4be33687aeba` |
| `tokenizer.json` | sha256 `5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3` |
| Vendored code | `src/gfmcd/adapters/dnabert2/vendor/` — see `VENDOR.sha256` and that directory's README |

## Deferred acquisitions

| Item | Needed for | Status |
|---|---|---|
| FANTOM5 CAGE peaks (hg38) | a frozen TSS shape/focus index — **blocker B1** | reachable, **not acquired** |
| EPDnew mouse + mm39/mm10 | the cross-species generality arm | **not acquired**; only mm9 is local, which is the wrong build |

Neither is promised by any gate in the current preregistration.
