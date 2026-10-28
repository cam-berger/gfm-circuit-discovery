# References

External works cited by [`docs/METHODOLOGY.md`](METHODOLOGY.md), including its [evidence ladder](METHODOLOGY.md#evidence-ladder).
The final block holds method-context works that no methodology document cites and that no claim ledger
carries.

**Verification status.** Every entry below was independently verified to exist *and* to state the specific
fact it is cited for, by retrieving the publisher or preprint record directly (arXiv abstract/full-text
pages, the JMLR landing page, the ACL Anthology entry, ICML/NeurIPS proceedings listings, Crossref
records) — the in-pipeline entries on **2026-07-22**, the method-context entries on **2026-07-31**.
References that could not be confirmed were omitted rather than approximated. Per house rule,
an unverified citation is worse than no citation: do not add an entry here without repeating that check,
and do not attach an existing entry to a statement it was not verified to support.

Citation keys are used in the methodology documents in the form `(Mueller et al., 2025)`.

---

## Mueller et al., 2025 — MIB

**Aaron Mueller, Atticus Geiger, Sarah Wiegreffe, Dana Arad, Iván Arcuschin, Adam Belfki, Yik Siu Chan,
Jaden Fiotto-Kaufman, Tal Haklay, Michael Hanna, Jing Huang, Rohan Gupta, Yaniv Nikankin, Hadas Orgad,
Nikhil Prakash, Anja Reusch, Aruna Sankaranarayanan, Shun Shao, Alessandro Stolfo, Martin Tutek, Amir Zur,
David Bau, Yonatan Belinkov.** *MIB: A Mechanistic Interpretability Benchmark.* International Conference on
Machine Learning (ICML) 2025. arXiv:2504.13151 — <https://arxiv.org/abs/2504.13151>

Establishes the field-standard formalism for reporting a whole faithfulness curve rather than a single
faithfulness number: **CPR** (Circuit Performance Ratio) is the area under the faithfulness-vs-circuit-size
curve, **CMD** (Circuit–Model Distance) is the area between that curve and `f = 1`, both approximated by the
trapezoidal rule over ten circuit sizes `k ∈ {.001, .002, .005, .01, .02, .05, .1, .2, .5, 1}`.

*Cited for:* the faithfulness-curve formalism in `METHODOLOGY.md` § "The faithfulness–size curve".

## Geiger et al., 2025 — causal abstraction

**Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaudhary, Sonakshi Chauhan, Jing Huang, Aryaman Arora,
Zhengxuan Wu, Noah Goodman, Christopher Potts, Thomas Icard.** *Causal Abstraction: A Theoretical Foundation
for Mechanistic Interpretability.* Journal of Machine Learning Research 26(83), 1–64, 2025.
<https://www.jmlr.org/papers/v26/23-0058.html> — also arXiv:2301.04709

Generalizes causal abstraction from mechanism replacement to arbitrary mechanism transformation, gives a
precise formalization of **graded faithfulness**, and explicitly unifies activation and path patching, causal
mediation analysis, causal scrubbing, causal tracing, circuit analysis, distributed alignment search and
steering within one causal-abstraction language.

*Cited for:* the causal-abstraction framing of "circuit" and of graded faithfulness in `METHODOLOGY.md`, including
its evidence ladder.

## Chan et al., 2022 — causal scrubbing

**Lawrence Chan, Adrià Garriga-Alonso, Nix Goldowsky-Dill, Ryan Greenblatt, Jenny Nitishinskaya,
Ansh Radhakrishnan, Buck Shlegeris, Nate Thomas.** *Causal Scrubbing: A Method for Rigorously Testing
Interpretability Hypotheses.* AI Alignment Forum (Redwood Research), 2022.
<https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing>

**Format caveat: this is a blog-format technical report, not a peer-reviewed publication.** Cite it as such.

Introduces causal scrubbing: an interpretability hypothesis is read as a claim about which internal
activations may be resampled without changing behavior, and is tested by performing exactly those
behavior-preserving resampling ablations and measuring how much behavior survives.

*Cited for:* the scrubbing comparison in `METHODOLOGY.md` § "Relation to causal scrubbing".

## Zhang & Nanda, 2024 — patching metrics and methods

**Fred Zhang, Neel Nanda.** *Towards Best Practices of Activation Patching in Language Models: Metrics and
Methods.* International Conference on Learning Representations (ICLR) 2024. arXiv:2309.16042 —
<https://arxiv.org/abs/2309.16042>

Systematically examines how the methodological details of activation patching — the evaluation metric and
the corruption method in particular — change localization and circuit-discovery results, shows that varying
these hyperparameters can yield disparate interpretability conclusions, and argues for specific choices.

*Cited for:* the statement that metric and corruption method are decision-critical axes, in `METHODOLOGY.md`
§ "Conformance with activation-patching best practice".

## Heimersheim & Nanda, 2024 — using and interpreting patching

**Stefan Heimersheim, Neel Nanda.** *How to use and interpret activation patching.* arXiv preprint, 2024
(no archival or peer-reviewed venue found). arXiv:2404.15255 — <https://arxiv.org/abs/2404.15255>

A practitioner guide to activation patching: what the different patching setups and directions license as
circuit evidence, and how metric choice affects interpretation.

**Scope caveat:** verified as a best-practices summary for applying and interpreting patching. An explicit
"necessity vs sufficiency" framing was **not** verified in the text — do not attach that exact wording to
this reference without checking the paper.

*Cited for:* practitioner guidance on what a given patching setup licenses, in `METHODOLOGY.md`
§ "Conformance with activation-patching best practice".

## Conmy et al., 2023 — ACDC

**Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso.**
*Towards Automated Circuit Discovery for Mechanistic Interpretability.* Advances in Neural Information
Processing Systems (NeurIPS) 2023 (Spotlight). arXiv:2304.14997 — <https://arxiv.org/abs/2304.14997>

Introduces ACDC, which automates the subgraph-identification step of circuit analysis by iteratively pruning
edges of the computational graph, and validates it by recovering previously hand-found circuits in GPT-2
Small (e.g. 5/5 component types for the Greater-Than task).

*Cited for:* the recommended-but-not-yet-run automated edge-attribution cross-check in `METHODOLOGY.md`
stage 4.

## Syed et al., 2024 — edge attribution patching

**Aaquib Syed, Can Rager, Arthur Conmy.** *Attribution Patching Outperforms Automated Circuit Discovery.*
Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages
407–416, 2024. ACL Anthology 2024.blackboxnlp-1.25, DOI 10.18653/v1/2024.blackboxnlp-1.25 —
<https://aclanthology.org/2024.blackboxnlp-1.25/> — also arXiv:2310.10348

Introduces edge attribution patching, a linear gradient-based approximation to activation patching that
scores every edge of the computational subgraph in two forward passes and one backward pass and then prunes
the least important edges, and reports that it outperforms existing automated circuit-discovery methods at
far lower cost. An earlier version appeared at a NeurIPS 2023 workshop; cite the archival BlackboxNLP 2024
version above.

*Cited for:* the recommended-but-not-yet-run automated edge-attribution cross-check in `METHODOLOGY.md`
stage 4.

---

**Method-context works — external, not ledger citations.** The entries below are external evidence *about
method*, preserved from a 2026-07-24 literature-verification pass (the source notes are retired). None is a
SpliceBERT result and none is entered in a claim ledger; entering one is the owner's step under the
immutable-record discipline. They support the behavior-first ordering — a behavior is causally qualified
before components are screened — and the argument that attention alignment is not causal importance.

## Kendiukhov, 2026 — single-cell FM interpretability audit

**Ihor Kendiukhov.** *Systematic Evaluation of Single-Cell Foundation Model Interpretability Reveals
Attention Captures Co-Expression Rather Than Unique Regulatory Signal.* arXiv preprint, submitted
19 February 2026. arXiv:2602.17532, DOI `10.48550/arXiv.2602.17532` — <https://arxiv.org/abs/2602.17532>

**Format caveat: single-author arXiv preprint, not peer-reviewed.** Cite it as such.

Thirty-seven analyses and 153 statistical tests over scGPT and Geneformer, four cell types and two
perturbation modalities. Attention patterns encode structured, layer-organized biology, yet that structure
provides no incremental value for perturbation prediction: trivial gene-level baselines outperform both
attention and correlation edges (AUROC **0.81–0.88** versus **0.70**), pairwise edge scores add zero
predictive contribution, and **causal ablation of regulatory heads produces no degradation**.

*Relevance:* the load-bearing exhibit for behavior-first ordering — attention alignment is not causal
importance.

## Kadem & Zheng, 2026 — attention-head intervention survey

**Mason Kadem, Rong Zheng.** *Interpreting Transformers Through Attention Head Intervention.* arXiv
preprint, submitted 7 January 2026 (v4, 26 February 2026). arXiv:2601.04398, DOI
`10.48550/arXiv.2601.04398` — <https://arxiv.org/abs/2601.04398>

**Format caveat: arXiv preprint, no archival or peer-reviewed venue found.** Cite it as such.

Traces how attention-head intervention emerged as a key method for causal interpretability, and frames the
evolution from visualization to intervention as a paradigm shift from observing correlations to causally
validating mechanistic hypotheses.

*Relevance:* the survey lineage for the visualization→intervention paradigm shift and the
specialization-versus-redundancy tension.

## Consens et al., 2025 — attention in genomic transformers

**Micaela E. Consens, Ander Diaz-Navarro, Vivian Chu, Lincoln Stein, Housheng Hansen He, Alan Moses,
Bo Wang.** *Interpreting Attention Mechanisms in Genomic Transformer Models: A Framework for Biological
Insights.* bioRxiv preprint, posted 27 June 2025. DOI `10.1101/2025.06.26.661544` —
<https://doi.org/10.1101/2025.06.26.661544>

**Format caveat: bioRxiv preprint, not peer-reviewed.** Cite it as such.

Offers a framework for reading attention in genomic transformer models toward biological insight.

*Relevance:* the genomic attention-to-biology direction that this method requires be causally validated
rather than trusted.

## Rood, Hupalowska & Regev, 2024 — perturbation cell and tissue atlas

**Jennifer E. Rood, Anna Hupalowska, Aviv Regev.** *Toward a foundation model of causal cell and tissue
biology with a Perturbation Cell and Tissue Atlas.* *Cell* 187(17), 4520–4545, August 2024. DOI
`10.1016/j.cell.2024.07.035` — <https://doi.org/10.1016/j.cell.2024.07.035>

Articulates the goal of a causal foundation model of cell biology built on a perturbation cell-and-tissue
atlas.

*Relevance:* the causal-foundation-model standard against which the single-cell audit reads as a gap
rather than a technicality.

---

**Biology-motivation works — external, not ledger citations.** The two entries below are the published
biology that motivated the **frozen positional contrasts** in
`studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md`
(§1). Neither supports any statement about SpliceBERT. They fix *which positions and which directions*
BG1 tests — the directions were taken from this literature before any model forward, not chosen from
model output — and nothing more. **No claim in any ledger cites them.** The BG1 development preflight
failed and BG1 validation/BG2 stayed sealed, so these works license no statement about SpliceBERT; see
`docs/NOMENCLATURE.md` (the `U2AF1` row, standing phrase 3) and
`docs/METHODOLOGY.md` § "3′-boundary organisation and U2AF1-centred site recognition" for the wording constraints that apply if they ever do.

Verified on 2026-08-10 by retrieving the Crossref DOI record and the open-access PMC full text for each,
per the house rule at the head of this file.

## Ilagan et al., 2015 — U2AF1 mutations and 3′ splice site preference

**Janine O. Ilagan, Aravind Ramakrishnan, Brian Hayes, Michele E. Murphy, Ahmad S. Zebari, Philip Bradley,
Robert K. Bradley.** *U2AF1 mutations alter splice site recognition in hematological malignancies.*
*Genome Research* 25(1), 14–26, January 2015. DOI `10.1101/gr.181016.114` —
<https://doi.org/10.1101/gr.181016.114> · PMID 25267526 · PMC4317169

Reports that U2AF1 mutations alter the preferred 3′ splice site motif **in patients, in cell culture, and
in vitro**, and shows that cassette exons with increased versus decreased inclusion carry different
consensus nucleotides at the **−3 and +1** positions.

*Cited for:* the existence of an experimentally resolved sequence organisation around the invariant `AG`,
in `PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md` §1. The −3/+1 positions in that protocol come from
here. **Verification note:** the −3/+1 detail is in the results, not the abstract.

## Yoshida et al., 2020 — U2AF1 structures and −3/+1 contacts

**Hisashi Yoshida, Sam-Yong Park, Gyosuke Sakashita, Yuko Nariai, Kanako Kuwasako, Yutaka Muto,
Takeshi Urano, Eiji Obayashi.** *Elucidation of the aberrant 3′ splice site selection by cancer-associated
mutations on the U2AF1.* *Nature Communications* 11, article 4744, 2020. DOI
`10.1038/s41467-020-18559-6` — <https://doi.org/10.1038/s41467-020-18559-6> · PMID 32958768 · PMC7505975

Crystal structures of wild-type and pathogenic-mutant U2AF1 bound to target RNA, with ITC binding
affinities. Resolves both contacts the protocol borrows: wild-type prefers **uracil at −3** (His29
stacking, Ser34 hydrogen bond) and **guanine at +1** (Phe165 π–π, Arg150 cation–π).

*Cited for:* the **direction** of the two frozen contrasts — −3 T-over-A and +1 G-over-A — in
`PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md` §1 and §4. This is what makes those directions
outcome-blind rather than data-selected.
