# Head–Context Atlas cohort v1

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Status: **outcome-independent mechanical preflight passed; no atlas model outcomes read**.

This document freezes the representation and matching policy for the first wider SpliceBERT atlas
screen. It does not freeze a head-function claim.

## Taxonomy policy

The six core labels are deliberately **overlapping and hierarchical**. They are properties of a
1024-nt packet and its local annotation graph, not mutually exclusive biological diagnoses.

| Label | Operational v1 meaning | Target roles screened |
|---|---|---|
| `single_pair` | exactly one fully visible donor–acceptor junction in the packet | donor, acceptor |
| `donor_cluster` | target donor belongs to a set of ≥2 donors sharing one acceptor | donor only |
| `acceptor_cluster` | target acceptor belongs to a set of ≥2 acceptors sharing one donor | acceptor only |
| `multiple_independent_pairs` | ≥2 visible junctions without a shared partner | donor, acceptor |
| `cassette_exon` | annotation graph contains the two inclusion edges and a skipping edge | donor, acceptor |
| `complex_event` | local graph carries more than one alternative/event relation | donor, acceptor |

`single_pair` is packet-scoped: the wider gene may contain alternatives outside the crop. `cassette_exon`
and `complex_event` may co-occur with clusters or with one another. Every non-focal core tag is matched
exactly, so a focal contrast is not silently a contrast in another event label.

The shared acceptor in a donor cluster and shared donor in an acceptor cluster are not assigned the same
focal label in v1. Those reverse-role views can be added later as separately named descriptive tags; they
must not be smuggled into the role-specific cluster estimand.

## Annotation-only census

Source: comprehensive GENCODE v50 + hg38. Sequence windows are transcript-oriented and canonical
GT/GC–AG only. chr8/9 (sealed) and chr11/16/19 (historically touched) are excluded before sequence
construction.

- 388,078 unique annotated junctions; 113,720 fit the 1024-nt coordinate contract.
- 227,440 donor/acceptor packets; 163,540 valid chromosome-train packets after sequence gates.
- 81,770 unique sequence windows; every masked target base is native `G`.
- A fresh, outcome-independent gene hash creates physically separate discovery and internal-validation
  files.

These counts establish representation only. They say nothing about head selectivity.

## Matching policy

Each focal context is contrasted with packets lacking that tag. Case and control:

- have the same target role, strand, exon/CDS region, distance modulo 3, 50-nt distance band, target-site
  transcript-support band, junction-support band, and all other core context tags;
- are propensity/caliper matched on continuous target/partner strength, distance, GC/pyrimidine content,
  motif burden, homopolymer, packet position, and ≈10.63-nt phase geometry;
- come from different genes; control reuse is capped at two within a contrast;
- must finish with maximum absolute numeric SMD ≤0.10 and exact non-focal tag balance.

Graph variables that define the focal context (`visible_pair_count`, donor/acceptor degree, local edge
count) are not adjusted away.

## Selected v7 cohort

The first six matching pilots are preserved as failed/superseded designs. v7 is the first candidate on
which every contrast passes the mechanical balance gate.

| Contrast | Discovery pairs | Internal-validation pairs |
|---|---:|---:|
| single pair — donor | 574 | 130 |
| single pair — acceptor | 575 | 107 |
| donor cluster — donor | 653 | 179 |
| acceptor cluster — acceptor | 640 | 158 |
| multiple independent pairs — donor | 575 | 172 |
| multiple independent pairs — acceptor | 638 | 208 |
| cassette exon — donor | 641 | 248 |
| cassette exon — acceptor | 711 | 321 |
| complex event — donor | 755 | 379 |
| complex event — acceptor | 725 | 345 |
| **Total** | **6,487** | **2,247** |

Discovery contains 11,807 unique packets / 10,900 sequence windows / 5,089 genes. Internal validation
contains 4,019 packets / 3,512 windows / 1,486 genes. The split gene and sequence sets are disjoint.

The validation minimum of 80 is a feasibility floor, not an equivalence license. In particular, the
single-pair validation arms (107/130 pairs) receive no bounded-null interpretation until the actual
head-effect variance and outcome-specific MDE are measured.

## Mechanical audit

`atlas_context_matched_v7.audit.json` independently rechecked all pair/packet/sequence joins, focal and
non-focal tags, role rules, different-gene matching, deterministic split assignment, excluded
chromosomes, masked-base coordinates, donor/acceptor motifs, sequence hashes, control reuse, recomputed
balance, summary counts, cross-split leakage, and every source-manifest hash.

Result: **PREFLIGHT_PASS; 0 failures**.

## Model preflight disposition

Cohort balance alone did not license Colab inference. The required small discovery-only mechanical model
preflight has now demonstrated:

1. donor masks `donor_win`; acceptor masks `acceptor_win-1`;
2. native masked MLM margins are finite and the source-to-source patch identity is exact;
3. all 96 heads and all six MLP hooks execute for both site roles;
4. masked non-site health probes distinguish target-specific contribution from broad MLM damage;
5. output rows carry packet, pair, context, role, split, checkpoint, and intervention provenance.

All gates passed: four packets, both roles, all labels, 816 component-estimand rows, exact source
identities, and usable masked health. The independent bulk code path reproduced 2,448 preflight values
with maximum absolute error 0.0, then passed shard validation and merge. The discovery screen is now
packaged for Colab; its own cross-device smoke, benchmark, and 512-packet masked-health panel remain hard
preconditions for the full target phase.
