# Pre-registration — the forward arm: does L3H1 deliver U_info? (donor-graph 1b)

**Status: FROZEN.** Written before `run_donor_forward.py` is executed. Definitions import from
`gfmcd.splicing.circuit` and reuse the anchor's `U_info`; no threshold below is revisable by the
run. Builds on **C16** (`PREREGISTRATION_DONOR_ANCHOR.md`): the coupling is carried by a low-rank
object `U_info` (top-SVD of the clean−corrupt final residual) that recovers 0.92 of the coupling.

## 0. The question, and the failure mode it guards

1a anchored the **output** end: `U_info`. `readout_path.json` established that restoring L3H1's
clean output into the corrupt run recovers most of the coupling through a distributed downstream
chain. 1b joins them: **is the margin that L3H1 restores carried by `U_info`, or does L3H1
deliver it through directions the anchor never saw?**

The trap: `U_info` = top-SVD(h_cl − h_co), and a full L3H1 restoration would make h_re ≈ h_cl,
whose U_info-component is clean by construction. That near-tautology is defused two ways: (a)
L3H1 restoration is **partial** (~0.82, not 1.0), so the test is whether the *part it restores*
lies in `U_info`; (b) the gate is licensed by a **size-matched random-subspace null** (a random
rank-8 subspace of h_re must route far less) — the control that can fail.

## 1. Target, corruption, restoration — imported unchanged

- Readout / corruption / coupling / eligibility: exactly as `CouplingTask` + C16 (val, both
  strands, gap [60,150], unconditional set, ratio-of-means recovery, gene-grouped `gboot_ratio`).
- **Restoration** = `patch_heads(ad, clean_head_outputs, [READER])` applied to the corrupt run,
  every downstream block **live** — the `readout_path` "all-pos, downstream free" intervention
  (a one-way copy of L3H1's clean output; not a swap).
- `U_info` = `top_k_subspace(h_cl − h_co, 8)`, recomputed here from the same final residuals.

## 2. Measurements

Cache per example: final donor residuals `h_cl`, `h_co`, `h_re` (restored) and margins
`m_cl`, `m_co`, `m_re`; and each downstream block's write at the donor
(`DOWN = [(3,mlp),(4,attn),(4,mlp),(5,attn),(5,mlp)]`) in the clean, corrupt, and restored runs.

- **Total L3H1 recovery** `R_L3 = ratio_of_means(m_re − m_co, m_cl − m_co)` (expected ≈ readout_path 0.82).
- **U_info-routed recovery** `R_U = ratio_of_means(margin(patch_component(h_co, h_re, U_info)) − m_co, m_cl − m_co)`.
- **Random-subspace null**: `R_U` for 200 random rank-8 subspaces of the residual.
- **F-Align (descriptive)**: `subspace_overlap(top_k_subspace(h_re − h_co, 8), U_info)` vs random.
- **Per-block assembly (descriptive, approximate)**: `proj_Uinfo(write_s)` at the donor for
  clean/corrupt/restored, per block — how each block writes into `U_info` (approximate: ignores
  inter-layer LayerNorm, so it is read qualitatively, not summed to the coupling).
- **Localization**: repeat `R_U` restoring L3H1 only at the donor vs only at donor ±5 vs elsewhere.

## 3. FROZEN gates

| Test | PASS threshold | What FALSIFIES it |
|---|---|---|
| **Precondition** | `R_L3 ≥ 0.50` | L3H1 restoration too weak to analyse — report and halt the routing gate. |
| **F-Route** (primary) | `R_U / R_L3 ≥ 0.80` **and** `R_U > max(200 random rank-8)`, add-one `p ≤ 0.05` | L3H1's restored margin flows mostly through directions **outside** `U_info` → the endpoint link is not `U_info` and the graph's output object is incomplete. |
| **F-Local** | `R_U` restoring at the donor ±5 ≥ 0.50 of `R_U` restoring everywhere | L3H1's U_info delivery is not localised to where it writes. |

`R_U/R_L3 ≥ 0.80` is reachable-falsifiable: L3H1 could raise the G-margin through a route the
anchor's `U_info` does not span (e.g., a second mechanism), giving `R_U ≪ R_L3`.

## 4. Firewall placement

Advances the donor–acceptor circuit's **mediation/path** account by one edge: it attributes the
anchor's output object `U_info` to L3H1's write. Level: OV/mediation → behavior (unchanged
overall level; a sharper localization of the established circuit). It does **not** resolve the
layer-4/5 attention into specific heads — that is 1c.

## 5. Post-hoc reflection (REQUIRED)

After the run, a `REFLECTION` block states: pass/fail per gate; whether `U_info` is confirmed as
the object L3H1 delivers (F-Route) and where the assembly happens (per-block); and — the intent
check — whether the forward arm now meets the backward arm at `U_info`, or whether a non-U_info
route means the anchor's object must be widened before tracing continues. A non-routing result is
a real finding about the graph, reported as such, not patched.
