# Third-party assets, access, and licenses

> **Tier: T1 — canonical.** Current disclosure for repository and paper artifacts.

The repository's original code and documentation are licensed under the root
[`LICENSE`](../LICENSE) (MIT). That license does **not** relicense model checkpoints, reference genomes,
annotations, human-subject data, or other third-party assets. Large models and source datasets are not
redistributed in the tracked repository; users obtain them from the upstream providers and must follow
their terms. The current manifest standard requires exact source and hash records, but historical coverage
is incomplete. Consult [`PREREG_CONFORMANCE.md`](PREREG_CONFORMANCE.md) and the relevant correction ledger
before treating any run as exactly reconstructible.

| Asset | Upstream source | Terms / license checked | Repository handling |
|---|---|---|---|
| SpliceBERT code and published checkpoints | [biomed-AI/SpliceBERT](https://github.com/biomed-AI/SpliceBERT) | BSD-3-Clause in the upstream repository | Checkpoints are downloaded separately and ignored by Git; this repository distributes only adapters, derived aggregate results, and hashes. |
| SpliceBERT GTEx analysis/fine-tuned checkpoints | [chenkenbio/SpliceBERT-analysis](https://github.com/chenkenbio/SpliceBERT-analysis) | BSD-3-Clause in the upstream repository | Checkpoints are not redistributed. No participant-level GTEx sequence or phenotype data is tracked. |
| GENCODE v50 and GRCh38 primary assembly | [GENCODE data access](https://www.gencodegenes.org/pages/data_access.html) | GENCODE states that its project data are open access | Source files are downloaded separately; manifests pin filenames and hashes. |
| GTEx data | [GTEx Portal](https://gtexportal.org/home/downloads/adult-gtex) and [dbGaP access guidance](https://www.grants.nih.gov/policy-and-compliance/policy-topics/sharing-policies/accessing-data/dbgap) | GTEx distinguishes open-access resources from protected participant-level data requiring authorization | This repository does not redistribute controlled GTEx data. Any future use of protected data requires independent authorization and compliance with its data-use limitations. |
| ENCODE data | [ENCODE data-use policy](https://www.encodeproject.org/about/data-use-policy/) | ENCODE releases data for unrestricted use immediately and requests appropriate citation | Source records are downloaded separately; only derived cohorts, aggregate results, and provenance are tracked. |
| DNABERT-2 source code | [MAGICS-LAB/DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2) | Apache-2.0 in the upstream repository | Source is not vendored. The base and promoter-fine-tuned checkpoints are downloaded separately. Their Hugging Face cards do not currently declare a model license, so the checkpoints are not redistributed under this repository's MIT license. |
| RNA-FM | [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM) | MIT in the upstream repository | Model files are downloaded separately and are not tracked. |
| RNA-BERT | Source cited by the RNA-BERT publication/model provider | No separate license was verified during this audit | Model files are not redistributed. Users must verify the current upstream terms before obtaining or sharing them. |
| EPDnew, PolyASite, FANTOM5, and other atlas catalogs | Provider named in each study manifest/protocol | Dataset-specific terms apply; no blanket relicensing is asserted here | These inputs are not redistributed. The manuscript's DNABERT-2 comparison is a bounded historical probe study, not a live claim from these catalogs. |
| NeurIPS 2026 author kit | Supplied NeurIPS 2026 formatting bundle | No separate license statement was present in the supplied `neurips_2026.sty` header | The exact style, example, and checklist used by the current draft are tracked under `tex/` for reproducible typesetting. Their inclusion does not relicense them under the repository's MIT license. |
| Third-party Python/R/system dependencies | `pyproject.toml`, lockfile, and environment documentation | Each package retains its upstream license | Dependencies are installed from their normal package channels; their source is not vendored by this repository. |

## Attribution and redistribution rules

- Cite the original dataset and model papers in publications, in addition to following provider-specific
  attribution instructions.
- Do not commit access tokens, participant-level controlled data, or checkpoint weights.
- Before publishing an artifact bundle, verify every included file against this table and its manifest.
- If an upstream model card lacks a declared license, do not infer one from the accompanying code
  repository; distribute only repository-owned code and independently derived aggregate results.

This table documents upstream statements and repository handling; it is not legal advice.
