"""Smoke tests for the interactive lab layer (gfmcd.lab).

These do not re-test the science -- signatures.py and oracle.py have their own suites. They test that
the interactive wrapper is safe and honest: a checkpoint loads, an ablation restores itself even when
the block raises, Δloss is what ablation reports, and each pillar's entry point returns a sane value
on a real checkpoint. The checkpoints they need (nano_uniform, repeat_copy_s0) are on disk; if they
are ever absent the tests skip rather than fail.
"""

import numpy as np
import pytest
import torch

import gfmcd.lab as lab
from gfmcd.signatures import start_index

pytestmark = pytest.mark.filterwarnings("ignore")


def _have(stem: str) -> bool:
    try:
        lab.resolve_checkpoint(stem)
        return True
    except FileNotFoundError:
        return False


needs_uniform = pytest.mark.skipif(not _have("nano_uniform"), reason="nano_uniform checkpoint absent")
needs_copy = pytest.mark.skipif(not _have("repeat_copy_s0"), reason="repeat_copy_s0 checkpoint absent")


@pytest.fixture(scope="module")
def rf():
    """The reading-frame counter and a small window batch."""
    m = lab.load_checkpoint("nano_uniform", device="cpu")
    tok, aa, phase = lab.load_uniform_windows(n=48, seed=555)
    start = start_index(phase)
    return m, tok, aa, phase, start


@pytest.fixture(scope="module")
def cp():
    """The repeat-copy induction model and a batch."""
    m = lab.load_checkpoint("repeat_copy_s0", device="cpu")
    batch = lab.load_repeat_windows(n=48)
    return m, batch


# ---- loading ---------------------------------------------------------------


@needs_uniform
def test_family_stem_resolves_to_latest_step(rf):
    m, *_ = rf
    assert m.name.startswith("nano_uniform_step")
    assert m.is_aa_head and not m.is_causal
    assert m.cfg.n_layers == 6 and m.cfg.n_heads == 8


@needs_uniform
def test_pin_step(rf):
    early = lab.load_checkpoint("nano_uniform", step=1200, device="cpu")
    assert early.step == 1200


def test_missing_checkpoint_names_families():
    with pytest.raises(FileNotFoundError) as e:
        lab.resolve_checkpoint("no_such_model_xyz")
    assert "families" in str(e.value).lower()


# ---- intervention safety ---------------------------------------------------


@needs_uniform
def test_ablation_restores_on_normal_exit(rf):
    m, *_ = rf
    before = float(m.model.blocks[1].attn.head_mask[3])
    with lab.ablate_heads(m.model, (1, 3)):
        assert float(m.model.blocks[1].attn.head_mask[3]) == 0.0
    assert float(m.model.blocks[1].attn.head_mask[3]) == before


@needs_uniform
def test_ablation_restores_after_exception(rf):
    """The reason the context manager exists: a raise inside must not leave a head zeroed."""
    m, *_ = rf
    before = float(m.model.blocks[1].attn.head_mask[3])
    with pytest.raises(RuntimeError):
        with lab.ablate_heads(m.model, (1, 3)):
            raise RuntimeError("boom")
    assert float(m.model.blocks[1].attn.head_mask[3]) == before


@needs_uniform
def test_resample_leaves_no_hook_behind(rf):
    """After a resample block, no instance-level forward override lingers and outputs are unchanged."""
    m, tok, *_ = rf
    attn = m.model.blocks[1].attn
    t = torch.from_numpy(tok[:8]).long()
    with torch.no_grad():
        before = m.model(t).clone()
    src = torch.from_numpy(tok[np.random.permutation(len(tok))]).long()
    with lab.resample_heads(m.model, [(1, 3)], src):
        pass
    assert "forward" not in vars(attn), "resample override was not removed from the instance"
    with torch.no_grad():
        after = m.model(t)
    assert torch.allclose(before, after), "forward not functionally restored after resample"


# ---- one assay per pillar --------------------------------------------------


@needs_uniform
def test_pillar1_attention_survey_shape(rf):
    m, tok, aa, phase, start = rf
    attn = lab.attention_survey(m, tok, start)
    assert attn.shape == (m.cfg.n_layers, m.cfg.n_heads)
    assert (attn >= 0).all()


@needs_uniform
def test_pillar2_ceiling_predicts_circuit(rf):
    """The counter beats the pooled ceiling, so a circuit is predicted."""
    m, tok, aa, phase, start = rf
    rep = lab.ceiling_report(m, tok, phase, demanded=1.0)
    assert rep.pooled_oracle_acc < rep.model_phase_acc
    assert rep.circuit_predicted


@needs_uniform
def test_pillar3_l1h3_essential_l1h2_inert(rf):
    """The load-bearing assertion: ablation, not attention, separates the two head-1 heads.

    L1H3 moves the loss by a lot; L1H2 (which attends the start codon with high specificity) is inert.
    """
    m, tok, aa, phase, start = rf
    tbl = lab.head_ablation_table(m, tok, aa, target_pos=start)
    d = {(e.layer, e.head): e for e in tbl}
    assert d[(1, 3)].delta_loss > 1.0
    assert d[(1, 2)].delta_loss < 0.01
    # and the table is sorted by Δloss
    assert tbl[0].delta_loss >= tbl[-1].delta_loss


@needs_uniform
def test_pillar4_pooled_ceiling_in_range(rf):
    m, tok, aa, phase, start = rf
    pooled = lab.pooled_ceiling(m, tok, phase)
    assert 0.3 <= pooled <= 1.0


@needs_uniform
def test_pillar5_counter_verdict(rf):
    m, tok, aa, phase, start = rf
    sig = lab.mechanism_signature(m, tok, phase)
    assert "counter" in sig.verdict
    assert sig.anchor_share > 0.5


# ---- the 2x2 loader and the output-level verdict (D35) ----------------------


def _cell_cache_exists(cell: str) -> bool:
    from gfmcd.lab.load import CACHE_DIR

    return any(CACHE_DIR.glob(f"cell{cell}_val_*_256_4242.npz"))


needs_cellC = pytest.mark.skipif(not _cell_cache_exists("C"), reason="no cell C window cache")


@needs_cellC
def test_load_cell_windows_is_cwd_independent_and_reuses_larger_cache(tmp_path, monkeypatch):
    """The notebook runs from notebooks/, where the training pool's relative cache misses. The lab
    loader must resolve the repo cache regardless of cwd, and reuse a larger-n cache when the exact
    n was never written (asking for more than any cache holds returns what exists, not an error)."""
    monkeypatch.chdir(tmp_path)  # a cwd with no data/cache at all
    tok, aa, phase = lab.load_cell_windows("C", n=10, seed=4242)
    assert tok.shape[0] == 10 and tok.shape[1] == 256
    assert aa.shape == tok.shape and phase.shape == tok.shape
    assert (phase >= 0).any()  # real coding windows, not empty


@needs_cellC
def test_verdict_2x2_reads_at_output_not_l2_probe(rf):
    """D35: on real CDS the anchor is not at L2, so the verdict must use the output-level anchor share.
    Cell C is a counter there (>0.5) while its L2 probe reads near zero -- the two must disagree, and
    the verdict must follow the output."""
    m, *_ = rf
    mc = lab.load_checkpoint("cellC_s0")
    tok, aa, phase = lab.load_cell_windows("C", n=200, seed=4242)
    v = lab.verdict_2x2(mc, tok, aa, phase, cell="C")
    assert v.anchor_share > 0.5           # output: a counter
    assert v.anchor_share_l2probe < 0.3   # L2 probe: the retracted reading
    assert v.circuit_formed               # verdict follows the output


# ---- repeat-copy family ----------------------------------------------------


@needs_copy
def test_copy_model_beats_base_rate(cp):
    m, batch = cp
    cm = lab.copy_metrics(m, batch)
    assert cm["copy_accuracy"] > 0.9
    assert cm["base_rate_ceiling"] < 0.35
    assert cm["conflict_follows_distal"] > cm["conflict_follows_local"]


@needs_copy
def test_copy_source_heads_are_layer1(cp):
    """The documented five source heads live in layer 1."""
    m, batch = cp
    attn = lab.copy_source_attention(m, batch)
    top = np.dstack(np.unravel_index(np.argsort(-attn.ravel()), attn.shape))[0][:5]
    layers = {int(l) for l, h in top}
    assert layers == {1}


@needs_copy
def test_copy_redundancy_is_partial(cp):
    """Sum of single-head effects far exceeds the joint effect (partial redundancy)."""
    m, batch = cp
    tbl = lab.copy_head_ablation(m, batch)
    heads = [(e.layer, e.head) for e in tbl[:5]]
    red = lab.copy_redundancy(m, batch, heads)
    assert red.subset_delta < red.sum_of_singles


# ---- reverse engineering ---------------------------------------------------


@needs_uniform
def test_phase_decodability_rises_with_depth(rf):
    """The frame becomes decodable across depth: low at L0, ~1.0 by mid-stack.

    This is the spatial signature the reverse-engineering section reads: the layer where the target
    becomes decodable is where the circuit finished computing it.
    """
    m, tok, aa, phase, start = rf
    decode = lab.phase_decodability_by_layer(m, tok, phase)
    assert len(decode) == m.cfg.n_layers
    assert decode[0] < 0.6          # not yet built at the first layer
    assert max(decode) > 0.95       # fully in place somewhere up the stack
    assert decode[-1] >= decode[0]  # monotone-ish: later layers carry it


def test_plot_circuit_flow_renders():
    """The flow diagram builds from a stage list without a live model."""
    import matplotlib
    matplotlib.use("Agg")
    from gfmcd.lab import viz

    stages = [
        {"layer": "L0", "role": "setup", "heads": "L0H0", "detail": "x", "kind": "setup"},
        {"layer": "L1", "role": "anchor", "heads": "L1H3", "detail": "y", "kind": "anchor"},
        {"layer": "out", "role": "readout", "heads": "", "detail": "z", "kind": "readout"},
    ]
    ax = viz.plot_circuit_flow(stages, probe_by_layer=[0.4, 0.9, 1.0])
    assert ax is not None
