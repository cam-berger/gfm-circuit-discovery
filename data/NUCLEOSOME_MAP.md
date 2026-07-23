# The Voong chemical-cleavage nucleosome map — provenance, and the traps in it

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Acquired 2026-07-13.** This is the external data Exp 2 (the dyad-anchored source assay) was
blocked on. It is now in hand.

## What was downloaded

| | |
|---|---|
| **Study** | Voong et al. 2016, *Cell* 167(6):1555–1570 — "Insights into Nucleosome Organization in Mouse Embryonic Stem Cells through Chemical Mapping" |
| **PMID** | [27889238](https://pubmed.ncbi.nlm.nih.gov/27889238/) |
| **GEO series** | [GSE82127](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE82127) |
| **Sample** | GSM2183909 (chemical cleavage, H4S47C) |
| **File** | `GSM2183909_unique.map_95pc.txt.gz` — the **unique dyad map at 95% confidence** |
| **Path** | `/Users/cberg/Research/data/nucleosome/GSM2183909_unique.map_95pc.txt.gz` |
| **Size** | 37,133,829 bytes |
| **sha256** | `f9b2c9e4dd90e9e63262f7cdfc69e252c6936e3d805bdb8d8e4294b87e813739` |
| **Dyads** | **10,677,053** |

⚠️ **Do NOT download `GSE82127_RAW.tar` (18.7 GB).** Everything else in the series is per-base
score tracks (`Chemical_NCPscore.txt.gz` alone is 3.9 GB). The dyad map is the *only* file the
assay needs, and it is 37 MB. The individual-file URL is:

```
https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2183nnn/GSM2183909/suppl/GSM2183909_unique.map_95pc.txt.gz
```

## ⚠️ TRAP 1 — the genome build is mm9, NOT mm10

The paper is from 2016 and the coordinates are **mm9**. Verified, not assumed: chr1's last dyad
sits at **197,195,276**, against **mm9 chr1 = 197,195,432** (mm10 chr1 = 195,471,971). It lands
156 bp short of the mm9 telomere — exactly where a 147-bp nucleosome's dyad must.

Using an mm10 FASTA would shift every dyad by a build-dependent, position-dependent offset —
a *silent* corruption that would look exactly like positional jitter, and would attenuate the
10.67-nt signal by `exp[−½(2πσ/T)²]` without ever raising an error. **The reference is
`/Users/cberg/Research/data/genomes/mm9.fa`.**

## ⚠️ TRAP 2 — the chromosomes are RENUMBERED, and 20/21/22 are not autosomes

The file's first column is a bare number with no `chr` prefix, and mouse has only 19 autosomes.
Confirmed against mm9 chromosome lengths:

| file | actual | max dyad | mm9 length |
|---|---|---|---|
| `1`–`19` | chr1–chr19 | — | — |
| **`20`** | **chrX** | 166,649,985 | 166,650,296 |
| **`21`** | **chrY** | 2,902,427 | 15,902,555 (mostly unmappable) |
| **`22`** | **chrM** | 16,106 | 16,299 |

Reading `20`/`21`/`22` as autosomes would index past the end of chr19 and either crash or, worse,
silently wrap. **chrM (37 dyads) must be excluded** — a 16 kb circular genome at high copy number
is not chromatin and will not carry a nucleosomal signal.

## Why this map, and not MNase-seq

MNase has a **~30× A/T cleavage bias** (Chereji 2019). The outcome of the assay *is* a weak
nucleotide periodicity, so an A/T-biased digestion would manufacture the very signal being tested
for. Chemical cleavage (H4S47C → hydroxyl radical at the dyad axis) has no such bias. This is
not a preference; it is a precondition — see D25.

## Power, against the frozen instrument

The assay (D25) needs **~3,000 well-positioned dyads** for 80% power at 2% modulation, and
**~13,000** at 2 bp of jitter. This map has **10.7 million** — 800–3,500× the requirement. The
binding constraint was never count; it is **precision** and the unknown **phase coherence ρ**.

Both are now measurable rather than assumed:
- **jitter** — the map's own 95%-confidence calling is the precision claim; the frozen-phase
  split-half will show whether it holds.
- **ρ = |E(e^{iφ})|** — cannot be simulated, only measured. If the phase estimated on chr1+2
  predicts the *signed* projection on chr3+4, ρ is high. That test is already frozen.

## What still has to be true for this to mean anything

A powered positive here establishes **provenance plausibility only** — that the biological signal
exists. It does **not** show SpliceBERT represents it. That requires Exp 3, the carrier × dyad-score
interaction. Both are needed before the carrier may be called biological.
