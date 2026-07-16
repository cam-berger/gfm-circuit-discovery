# Decisions — acceptor-address transplantation (donor-graph experiment 3)

Chronological log of design changes made *before any confirmatory run* (val/test unseen),
recording what the setup-time diagnostics revealed and how the frozen protocol was corrected.
Per METHODOLOGY.md, corrections live here; the prereg is edited only pre-run with no data seen.

## 2026-07-15 — setup diagnostics (n=48 val, DIAGNOSTIC ONLY via --force, non-confirmatory)

Two principled problems surfaced on first contact with the model; both were anticipated risks
(R15; distance/motif salience) and are now fixed in the frozen protocol before any gate is read.

1. **The key+value transplant is R15-forced (position-insensitive).** Relocating L3H1's
   clean partner key+value recovered the coupling *equally* at the decoy, a non-site, and back
   at a+ (0.627 / 0.628 / 0.654). Relocating a high-affinity KEY draws the donor query to
   wherever it is placed and the co-relocated VALUE is read there — architecturally forced,
   exactly R15. Consequence: the key+value transplant CANNOT demonstrate content-addressing.
   The centre of gravity moves to (a) the partner-identity PREFERENCE and (b) the key-only
   address<->content DISSOCIATION (key-only redirects attention, S_A'=+0.47, but does not
   recover behaviour, 0.139; value-only ~0). The position-insensitivity of key+value is now
   REPORTED as the R15 guard, not gated as if it could be specific.

2. **The random-AG decoy is salience-confounded, and cannot be de-confounded by matching.**
   G-Match motif gap was ~1.3 s.d.: an intronic AG is a far weaker 3' splice site than the true
   partner, so the generic-AG preference is partly "L3H1 prefers stronger splice sites," not
   partner identity. Motif-closest selection did NOT close the gap (an intronic AG has no real
   3'SS to match) and it worsened the distance match. Resolution (final): the generic-AG decoy
   is matched on **distance + reading phase only** (a "partner vs generic matched site"
   comparison; motif_gap is reported, not gated), and the **strength-controlled partner-IDENTITY
   claim is carried by a non-partner REAL-acceptor decoy** (`real_acceptor_decoy`), where 3'SS
   strength is matched by construction (both are genuine acceptors). At n=60, S_A over the real
   acceptor was +0.59 (n=11 with such a decoy) — the identity signal survives strength control,
   though the real-acceptor arm has modest power (~18% of pairs have one) and is reported with
   that caveat.

Machinery confirmed correct at setup: swap-is-identity 2.9e-6 (R7 tripwire live), partner
coupling 1.00, L3H1 mean-ablation kills 74%, masked-probe health >0.5 under ablation, eager
attention rows sum to 1. The instrument works; the two fixes above are design corrections, not
bug fixes.

## 2026-07-15 — adversarial verification round (5 lenses x find/verify, pre-run)

A multi-agent adversarial review of the frozen setup (code + prereg) confirmed the core is
correct (the QK readout softmax(q.k/sqrt32) reproduces the model's own L3H1 attention to 1.9e-9;
head 1 = slice [32:64]; scaling 1/sqrt32; R7 one-way-copy discipline; kv demoted to an R15
guard) and surfaced defects, all fixed pre-run:

* **HIGH — real-acceptor identity arm distance-confounded/unaudited.** Non-partner real acceptors
  sit farther from the donor than the intron-mate partner, so S_A_real>0 could be attention
  distance-decay. Fix: `real_acceptor_decoy` takes `--real-tol`; `match_audit` reports the SIGNED
  gap distribution (dist_real-dist_plus); the result JSON carries it; §7.2 must adjudicate
  identity-vs-distance with it.
* **MED — C-Redirect corruption-confounded.** S_A'=A(a-)-A(a+) on the corrupt run is positive
  even with no relocation (a+ destroyed). Fix: gate on the difference-in-differences rise at the
  decoy, A(a-)[+key]-A(a-)[no reloc].
* **MED — --force wrote status=CONFIRMATORY on a failed preflight and could clobber the canonical
  artifact.** Fix: status = CONFIRMATORY iff (preflight_PASS and n>=250) else DIAGNOSTIC; manifest
  filename tracks --out; result JSON carries confirmatory/meets_power_target/preflight_PASS.
* **MED — prereg §0 re-committed the R15 "content-addressed read" term.** Fix: restated
  mechanism-neutrally via an inline pre-run amendment.
* **LOW — C-Dissociate gate embedded the R15-forced kv>k (cannot fail).** Fix: gate only the
  falsifiable legs (k lo>0, k>v, v<=0.15); kv>k reported as R15-expected.
* **LOW — overall verdict ANDed in G_Match (prereg excludes it).** Fix: overall = G_Preference
  and C_Redirect and C_Dissociate; G_Match reported.
* **LOW — C-Address 95-head null included ~48 structurally-inert layer>=3 heads** (S_A=attn[3]
  depends only on layers 0-2), deflating add-one p. Fix: restrict the null to layers 0-2 (47 heads).
* **LOW — provenance/robustness:** manifest now hashes GTF+FASTA; added a G-QK reconstruction
  preflight invariant (softmax(q.k/sqrt32)==model attn <1e-3); n_genes recorded; gate the
  systematic distance BIAS (|Δ of means|<=8) while reporting the per-pair spread (loose by AG
  sparsity, ~20-30nt).

NOT fixed (out of scope / frozen, pre-existing, not introduced here): the shared frozen
`circuit.py` labels READER "the content-addressed read" and `relocate_proj`'s docstring states the
R15 inference as a conclusion; `run_sealed_test.py` arm D still prints R15-retracted "content-
addressed" prose. These live in signed/hash-frozen code and belong to the sealed-test thread; they
should be addressed there (superseding note), not by editing frozen modules from this experiment.

## 2026-07-16 — confirmatory run (val, n=260, 215 genes, both strands) — PASS

Frozen protocol executed on val after the setup was committed (git 7aa47a0). Preflight 13/13 PASS
(injection 0.62, value_only 0.04, swap 3.3e-6, qk_recon 4.8e-7, dist_bias<=8, phase 0). All §4
gates PASS; `overall=True`; status CONFIRMATORY.

* **Preference:** S_QK +3.60 [3.37,3.80], S_A +0.55 [0.51,0.59], S_Y +0.87 [0.75,0.98].
* **Identity (real-acceptor arm, salience-controlled):** S_A +0.62 [0.46,0.74] (n=30). Distance
  confound adjudicated (prereg §7.2): real decoys median +19.5nt farther, but OLS intercept at
  matched distance +0.61 +-0.08, slope +0.001/nt, corr +0.08 — the identity effect is NOT
  distance-decay. This is the novel R15-safe result over the restore-at-position C9.
* **Address<->content:** C-Redirect (no-reloc DiD) +0.40 [0.37,0.44]; C-Dissociate key 0.16
  [lo 0.12] > value 0.03. C-R15: kv 0.62/0.62/0.65 (spread 0.024) — position-forced, reported.
* **C-Address (reported):** L2H15 retains 0.89 of S_A ablated (removes 11%), p=0.021 vs 47 upstream
  heads — a real but minor builder; the matched-decoy preference is built diffusely.

Registered as CLAIMS.md C20. Reflection filled in acceptor_address.json (prereg §7). Determinism
confirmed: an independent recomputation of the real-acc arm reproduced S_A +0.616 exactly.
Not run: the sealed test (chr8/9) remains available as an out-of-sample replication if desired.
