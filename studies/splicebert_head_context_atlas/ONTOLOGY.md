# ONTOLOGY — SpliceBERT Head–Context Atlas (v1, FROZEN)

> Frozen 2026-07-18 after the 6-lens design red-team (`REDTEAM_SYNTHESIS_C_6lens.md`). Annotation-only
> for v1. Model-derived quantities live in a SEPARATE namespace (§H) and MUST NOT define a biological
> class (R22). Amendments allowed only pre-run, logged inline as `Amendment (date, pre-run, no data seen)`.

The atlas annotates at **four linked scopes**; a tag names the scope it lives at. This prevents the
category error of calling a packet `single_site` because a partner fell outside the crop.

| Scope | Unit | Examples |
|---|---|---|
| **packet** | one 1024-nt model window | `partner_in_window`, `edge_proximal`, visible donor/acceptor counts |
| **site** | one donor or acceptor | `annotated_site`, MaxEnt strength, `n_annotated_transcripts_for_site` |
| **pair / event** | one intron-mate junction or competition event | `alt5ss_donor`, `alt3ss_acceptor`, `n_annotated_transcripts_for_junction` |
| **position** | one nucleotide | donor-core / acceptor-core base, CDS codon slot, exon/intron |

---

## The competition unit (CORRECTED — intron-mate, not exon-mate)

A **junction** is one annotated intron linking consecutive transcript-order exons `(e_k, e_{k+1})`:
`donor = _donor_anchor(e_k)` (5′SS), `acceptor = _acceptor_anchor(e_{k+1})` (3′SS). This is the
relationship `donor_margin` and the C1–C35 circuit operate on. (The first draft used
`partner_data.attested`, which is **exon-mate** alt-5′ — a different splice relationship; the red-team
killed it. Source of truth: `src/gfmcd/splicing/competition.py`.)

- **`alt5ss_donor`** (donor competition): ≥2 distinct donors splice to ONE shared downstream acceptor.
  target = a donor `t`; shared partner = the acceptor `p`; competitor = another donor `c` → `p`.
- **`alt3ss_acceptor`** (acceptor competition): ≥2 distinct acceptors receive ONE shared upstream donor.
  target = an acceptor `t`; shared partner = the donor `p`; competitor = another acceptor `c`.
- Competitors sharing a partner necessarily come from **different transcripts** ⇒ mutually exclusive by
  construction.

### Competition-topology fields (§B)
| field | scope | meaning / rule |
|---|---|---|
| `cluster_size` | event | # competing same-type sites for the shared partner |
| `n_usable_competitors` | event | competitors in-window AND ≥ `min_separation` from the target |
| `nearest_competitor_distance` | event | bp to nearest competitor (tandem-aware; exact, never binarised) |
| `target_competitor_distance` | event | bp target↔primary competitor |
| `target_partner_distance` | event | bp target↔shared partner (≈ intron length; the main window-fit constraint) |
| `is_tandem` | event | nearest competitor < 30 nt (NAGNAG / tandem wobble) — **EXCLUDED** from the primary cohort |
| `local_degree` | site | # splice-graph edges incident on the site (continuous complexity) |
| `min_separation` | config | frozen 30 nt: primary cohort requires target↔competitor ≥ 30 nt (drops NAGNAG/tandem) |
| `in_window` / `partner_in_window` | packet | target, partner AND competitor all fit the 1024-nt window (`span ≤ 824`) |
| `edge_proximal` | packet | any of the three within 4 nt of the crop boundary |

**Scope caveat (frozen):** only ~10% of annotated competition targets fit a 1024-nt window (median intron
≈3.8 kb; census v1). The primary cohort is therefore explicitly scoped to **competition resolvable within
1 kb** — a short-intron / proximal-alt-site sub-population. Cross-site-type frequency comparisons are
flagged **distance-truncation-confounded** unless window-inclusion rates are equalised.

---

## §A Site inventory & packet geometry (annotation)
`donor_count`, `acceptor_count`, `pair_count` (fully-visible junctions), `single_pair`, `multi_site`,
`multiple_independent_pairs` (pairs with no shared partner — NOT competition), `partner_in_window`,
`edge_proximal`. `multi_site` is a descriptive parent and **must not imply competition**.

## §C Splicing event (annotation-derived — names carry the evidence caveat)
`annotation_constitutive` (junction in all relevant annotated transcripts — NOT proven constitutive in
tissue), `alternative_donor`, `alternative_acceptor`, `cassette_exon`, `mutually_exclusive_exon`,
`intron_retention`, `alternative_first_exon`, `alternative_last_exon`, `complex_LSV` (>2 local
alternatives). v1 primary cohort uses `alt5ss_donor` / `alt3ss_acceptor` only; the rest are census tags.

## §D Site identity & EVIDENCE tier (this is where the data limit bites)
| tag | v1 status | reason |
|---|---|---|
| `annotated_site` | **POPULATED** | GENCODE v50 comprehensive |
| `matched_decoy` | **POPULATED** | real, non-partner, MaxEnt-strength+distance-matched to the COMPETITOR (MF6) |
| `unannotated_candidate` | **POPULATED** | motif scan + MaxEnt; no expression claim |
| `n_annotated_transcripts_for_site` | **POPULATED** | per-site transcript-support count (renamed from the misleading "junction_supported"; MF/correction 1) |
| `n_annotated_transcripts_for_junction` | **POPULATED** | per-junction transcript-support count |
| `junction_observed` | **DEFERRED → v2** | needs recount3/intropolis RNA-seq junctions (absent on disk) |
| `expressed_unannotated` | **DEFERRED → v2** | needs junction-read evidence |
| `activated_cryptic` | **INSTRUMENT-LIMITED → v3** | needs condition-/variant-linked evidence; a junction catalogue alone cannot establish activation (correction 2) |

## §E Junction chemistry & local strength (annotation + MaxEnt as STRATIFIER only, R22)
`GT_AG` / `GC_AG` / `AT_AC` / `noncanonical`; U2/U12 where supported; raw MaxEnt `score5`(donor)/
`score3`(acceptor); score quantiles; strength rank within the local competition set; strength gap to the
next-strongest; # stronger competitors. **MaxEnt is an external covariate/stratifier, never semantic
ground truth (R22).**

## §F Transcript & coding context (annotation)
GENCODE biotype (protein_coding / lncRNA / NMD / retained_intron / …); MANE Select vs alternative;
CDS / 5′UTR / 3′UTR / noncoding; CDS phase + codon slot (`phase.codon_slot`); frame-preserving vs
frameshifting alternative; first / internal / terminal; strand. v1 primary cohort = protein_coding.

## §G Model-independent sequence covariates (CONFOUND CONTROLS, not cohort-defining)
GC & pyrimidine content, k-mer frequencies, low-complexity / homopolymer, repeat overlap, local
donor/acceptor motif burden (in-window GT/AG counts), site-to-partner distance, competitor strength,
packet position / available flank. Used to adjust `E_ih`, never to define a class.

## §H Model-derived properties — SEPARATE NAMESPACE (never define a biological class; R22)
`native_base_logprob`, `output_entropy`, `dag_screening_attribution`, `exact_head_effect` (`E_ih`),
`M_comp` (behavioral competition dependence), `B` (partner coupling), P3 / ≈10.63 carrier
projection·amplitude, QK-selection & OV-writing main effects, `U_info`/`U_read` alignment, group /
faithfulness-size statistics. **A biological class is NEVER defined using the same SpliceBERT measurement
later treated as its explanation.**

---

## Frozen determinism & provenance
`sorted(set(...))` at every hashable-iteration site; `PYTHONHASHSEED=0`; a cross-seed determinism test.
Every artifact carries an `ExperimentManifest` pinning the checkpoint sha, GTF sha
(`83fba3e9…13a453`, comprehensive v50), seeds, and code revision.

---

## Amendment — 2026-07-19, pre-run for the wider atlas

No wider-atlas model outcomes had been read when this amendment was made. The competition-specific
ontology above remains historically frozen for that branch. For the wider Head–Context Atlas:

- the core context labels are overlapping/hierarchical multi-hot tags, not mutually exclusive cohorts;
- `single_pair` means one fully visible junction **in the packet**, not a globally single-junction gene;
- `donor_cluster` is a focal donor-role contrast and `acceptor_cluster` a focal acceptor-role contrast;
- cassette and complex labels may overlap with clusters, and every other core label is matched exactly;
- thin internal-validation strata may be called exploratory, but neither coordinate nor balance gates are
  weakened.

The operational definitions, counts, selected contrasts, and claim boundary are frozen in
`ATLAS_COHORT_V1.md` and `atlas_context_matched_v7.audit.json`.
