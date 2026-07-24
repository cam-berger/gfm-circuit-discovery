# Pre-registration — resolving L4/L5 attention into heads (donor-graph 1c)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_donor_heads.py` is executed. Definitions import from
`gfmcd.splicing.circuit` / `gfmcd.circuits.hf_bert`; no threshold is revisable by the run.
Builds on **C17** (L3H1 delivers `U_info`, assembled at MLP3/MLP4) and the `readout_path`
finding that the layer-4 and layer-5 **attention** stages contribute to the coupling recovery
as whole blocks. 1c resolves those two blocks into individual heads.

## 0. The question and its failure mode

L3H1 writes the acceptor-dependent signal at donor±5; the readout is at the donor. Something
must carry it across positions and layers. `readout_path` credited the L4/L5 attention blocks;
1c asks **which heads**, and **how** — do they route (QK) or transport content (OV/value)?

The trap to avoid: **mean-ablation** removes a head entirely and so flags heads that are merely
*present* on the pathway, not heads that carry the *acceptor-dependent* signal (R12/R13:
destroying structure ≠ measuring it). 1c uses **noising-to-corrupt** instead — replace a head's
clean output with the value it takes when the acceptor is destroyed — which changes only the
acceptor-dependent part of its output. A head whose noising leaves the coupling intact does not
carry the signal, however "important" it looks under ablation.

## 1. Target — imported unchanged

Readout / corruption / coupling / eligibility exactly as C16–C17 (val, both strands, gap
[60,150], unconditional set, ratio-of-means, gene-grouped `gboot`/`gboot_ratio`). Coupling
`coup = margin(clean) − margin(corrupt)` per example.

## 2. Part 1 — head necessity scan (noising)

Cache the **corrupt** per-head outputs (`capture_head_outputs` on the corrupt run). For each head
`h` in the **32 layer-4 and layer-5 heads**, plus reference heads **L3H1, L2H15, L1H12**:

- Run the **clean** forward with `patch_heads(ad, corrupt_cache, [h])` — h's output set to its
  corrupt value; everything else clean.
- `damage(h) = ratio_of_means(m_clean − m_clean|h→corrupt, coup)` = the fraction of the coupling
  that h's acceptor-dependent output carries, with a gene-grouped bootstrap CI.

## 3. Part 2 — router vs transporter (top heads)

For the **top 3 layer-4/5 heads by `damage`**, split the intervention at all nucleotide
positions (`patch_proj` with the corrupt projection as source, in the clean run):

- `damage_QK(h)` — noise h's **query and key** to corrupt (disrupt where it reads).
- `damage_V(h)` — noise h's **value** to corrupt (disrupt what it reads).

Classify: **router** if `damage_QK ≥ damage_V` by ≥0.05; **transporter** if `damage_V >
damage_QK` by ≥0.05; **mixed** otherwise. (Descriptive; not a pass/fail gate.)

## 4. FROZEN gates

| Test | PASS threshold | What FALSIFIES it |
|---|---|---|
| **Sanity** | `damage(L3H1)` is the largest of all scanned heads, and `damage(L3H1) > damage(L2H15) > 0` | the estimator does not recover the known reader/addresser — the scan is not trustworthy, halt. |
| **F-Heads** (primary) | at least one layer-4/5 head has `damage` CI-low **> 0.05** | no L4/L5 head individually carries ≥5% of the coupling → the relay is **distributed across heads** (a real finding, reported as C7-style head-level incompleteness, not forced). |
| **F-Selective** | the noising is head-selective: relay heads (damage CI-low > 0.05) are a **minority** (≤ ⅓) of the 32 L4/L5 heads | most heads' noising destroys the coupling → the metric reflects global breakage, not specific relays. |

**Amendment (2026-07-15, pre-run, before the n≥120 analysis).** The original F-Health gate
required `mlm_health > 0.95` under mean-ablation of each relay head. That is the wrong control:
mean-ablation removes a head *entirely*, whereas the damage metric uses gentle acceptor-specific
**noising** (clean→corrupt value, which differs only near the corruption). A general-purpose head
that *also* carries the coupling fails full-ablation health while its noising is perfectly
specific — so full-ablation health conflates "dedicated splice head" with "carries the signal."
The aligned specificity control is the **scan's selectivity** (a global-breakage artifact would
damage the coupling under *every* head's noising, not a specific few), promoted to **F-Selective**
above. Mean-ablation health is still computed and reported **descriptively**, to characterize
whether the relays are dedicated or general-purpose heads — not as a pass/fail gate.

## 5. Firewall placement

Sharpens the established donor–acceptor mediation/path account: identifies the specific L4/L5
heads that relay `U_info` from donor±5 to the donor and whether they route or transport. Level
unchanged (mediation → behavior); no new promotion. The whole-block readout_path result is not
revised, only decomposed.

## 6. Post-hoc reflection (REQUIRED)

State per-gate pass/fail; the identified relay heads and their router/transporter class; whether
the L4/L5 relay is head-sparse or distributed; and the intent check — does 1c close the
meet-in-the-middle (L3H1 → MLP3/MLP4 assemble → **these heads route/transport** → U_info →
logit), or does a distributed/absent result mean the attention stages resist head-level
resolution. A distributed result is a real finding, not a failure to be patched.
