"""One-call loading for interactive reverse-engineering.

The scripts each re-implement the same three lines to open a checkpoint
(`torch.load` -> `NanoConfig(**blob["cfg"])` -> `load_state_dict`) and the same
`.npz` cache convention to get windows. This module makes both a single call, so
a notebook opens with

    m = load_checkpoint("checkpoints/nano_uniform_step12000.pt")
    tok, aa, phase = load_windows("A")

and nothing else. It adds no science; it wraps what `scripts/train_cell.py::pool`
and every `scripts/*.py` main() already do, so the notebooks and the CLIs load a
model the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gfmcd.models.gfm_nano import GFMNano, NanoConfig

# Repo root, from src/gfmcd/lab/load.py -> ../../..
REPO = Path(__file__).resolve().parents[3]
CKPT_DIR = REPO / "checkpoints"
CACHE_DIR = REPO / "data" / "cache"


def pick_device(prefer: str | None = None) -> str:
    """mps if available else cpu, unless overridden. Matches every script's choice."""
    if prefer is not None:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LoadedModel:
    """A checkpoint, opened. `model` is in eval() mode on `device`.

    `pool` records which data regime produced it (uniform / cellC / repeat_copy / ...),
    read straight from the checkpoint blob so an assay can branch on it without the
    caller having to remember. `head_family` is a convenience label for the plot legends.
    """

    model: GFMNano
    cfg: NanoConfig
    device: str
    step: int | None
    pool: str | None
    path: Path
    meta: dict = None  # any extra blob fields: lam, shortcut_nats, cell, seed, ...

    @property
    def is_aa_head(self) -> bool:
        """True for the amino-acid-readout models (forcing task / 2x2 cells)."""
        return self.cfg.out_dim == 21

    @property
    def is_causal(self) -> bool:
        return self.cfg.causal

    @property
    def name(self) -> str:
        return self.path.stem

    def __repr__(self) -> str:
        head = "aa(21)" if self.is_aa_head else f"lm({self.cfg.out_dim})"
        arm = "causal" if self.cfg.causal else "mlm"
        return (
            f"LoadedModel({self.name}, pool={self.pool}, step={self.step}, "
            f"{arm}, head={head}, {self.cfg.n_layers}L x {self.cfg.n_heads}h, device={self.device})"
        )


def _latest_step(paths: list[Path]) -> Path:
    """Pick the highest-step checkpoint from a `<stem>_step<N>.pt` family."""
    def step_of(p: Path) -> int:
        stem = p.stem
        return int(stem.rsplit("_step", 1)[1]) if "_step" in stem else -1

    return max(paths, key=step_of)


def resolve_checkpoint(path_or_name: str | Path, step: int | None = None) -> Path:
    """Accept a full path, a path relative to repo root, a bare `.pt` name, or a family stem.

    A family stem like "nano_uniform" resolves to the latest `nano_uniform_step<N>.pt` on disk
    (or the given `step`). This is what makes `load_checkpoint("nano_uniform")` just work in a
    notebook without the caller memorising step numbers.
    """
    p = Path(path_or_name)
    direct = [p, REPO / p, CKPT_DIR / p.name]
    if p.suffix != ".pt":
        direct.append(CKPT_DIR / f"{p.name}.pt")
    for c in direct:
        if c.exists():
            return c

    # Family stem: nano_uniform -> nano_uniform_step*.pt
    stem = p.name[:-3] if p.name.endswith(".pt") else p.name
    if step is not None:
        exact = CKPT_DIR / f"{stem}_step{step}.pt"
        if exact.exists():
            return exact
    family = sorted(CKPT_DIR.glob(f"{stem}_step*.pt"))
    if family:
        return _latest_step(family)

    families = sorted({q.stem.rsplit("_step", 1)[0] for q in CKPT_DIR.glob("*_step*.pt")})
    raise FileNotFoundError(
        f"no checkpoint '{path_or_name}'. Known families: {families}. "
        f"Pass a family stem, a full name, or step= to pin a step."
    )


def load_checkpoint(path_or_name: str | Path, step: int | None = None, device: str | None = None) -> LoadedModel:
    """Open a checkpoint into an eval-mode GFMNano. The blob schema is {cfg, state, step, pool}.

    `path_or_name` may be a full path, a `.pt` filename, or a family stem ("nano_uniform"); a stem
    resolves to the latest step unless `step` pins one.
    """
    path = resolve_checkpoint(path_or_name, step=step)
    device = pick_device(device)
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = NanoConfig(**blob["cfg"])
    model = GFMNano(cfg).to(device).eval()
    model.load_state_dict(blob["state"])
    meta = {k: v for k, v in blob.items() if k not in ("cfg", "state")}
    return LoadedModel(
        model=model,
        cfg=cfg,
        device=device,
        step=blob.get("step"),
        pool=blob.get("pool"),
        path=path,
        meta=meta,
    )


def list_checkpoints(families: bool = True) -> list[str]:
    """Checkpoint families ("nano_uniform", "repeat_copy_s0", ...) or every stem, for notebook discovery."""
    if families:
        return sorted({p.stem.rsplit("_step", 1)[0] for p in CKPT_DIR.glob("*_step*.pt")})
    return sorted(p.stem for p in CKPT_DIR.glob("*.pt"))


# ----------------------------------------------------------------------------
# Windows. Amino-acid / phase-labelled windows for the reading-frame models.
# ----------------------------------------------------------------------------


def _find_cached(cell: str, split: str = "val") -> Path | None:
    """The largest cached val window set for a cell, if any (avoids regenerating hg38)."""
    hits = sorted(
        CACHE_DIR.glob(f"cell{cell}_{split}_*.npz"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return hits[0] if hits else None


def load_windows(
    cell: str = "A",
    n: int = 300,
    window: int = 256,
    seed: int = 99,
    prefer_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tokens, amino-acid labels, phase labels) for a reading-frame cell.

    Cells A/B are synthetic and regenerate deterministically. Cells C/D need hg38 and are
    served from `data/cache/*.npz` when present (hg38.fa itself need not be installed).

        A   uniform codons (the forcing task; the anchored counter)
        B   uniform + decoy ATGs + planted Kozak (hard detector)
        C   real hg38 CDS, no upstream ATG
        D   real hg38 CDS, natural
    """
    cell = cell.upper()
    if prefer_cache:
        cached = _find_cached(cell)
        if cached is not None:
            z = np.load(cached)
            return z["tok"], z["aa"], z["ph"]

    if cell == "A":
        from gfmcd.data.uniform_cds import make_batch

        return make_batch(n, window, seed=seed)
    if cell == "B":
        from gfmcd.data.uniform_cds import make_batch_hard_detector

        return make_batch_hard_detector(n, window, seed=seed)
    raise FileNotFoundError(
        f"cell {cell} needs hg38 and no cache was found in {CACHE_DIR}. "
        "Cells C/D require data/cache/cell{C,D}_val_*.npz (or hg38.fa + GENCODE to regenerate)."
    )


def load_uniform_windows(
    n: int = 384, window: int = 256, seed: int = 555
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fresh uniform-codon windows (tok, aa, phase). The forcing-task substrate."""
    from gfmcd.data.uniform_cds import make_batch

    return make_batch(n, window, seed=seed)


def load_repeat_windows(n: int = 256, window: int = 256, seed: int = 4242):
    """A RepeatCopyBatch for the induction models. See gfmcd.data.repeat_copy.RepeatCopyBatch."""
    from gfmcd.data.repeat_copy import make_batch

    return make_batch(n, window, seed=seed)


def load_history(stem: str) -> list[dict]:
    """The training history rows for a checkpoint family: [{step, train, val/loss, aa_acc/copy_acc}, ...].

    This is the grokking curve -- loss and accuracy at every logged step. Written by the training
    scripts as `checkpoints/<stem>_history.json`.
    """
    import json

    stem = stem[:-3] if stem.endswith(".pt") else stem
    stem = stem.rsplit("_step", 1)[0]
    path = CKPT_DIR / f"{stem}_history.json"
    if not path.exists():
        raise FileNotFoundError(f"no history at {path}. Families with history: "
                                f"{sorted(p.stem[:-8] for p in CKPT_DIR.glob('*_history.json'))}")
    return json.loads(path.read_text())


def grok_transition(history: list[dict], acc_key: str) -> dict:
    """Locate the grokking transition: the step interval of steepest accuracy rise, plus threshold crossings.

    Returns {transition_step, transition_from, transition_to, step_at_0.5, step_at_0.9, final_acc}.
    The transition step is the later endpoint of the steepest-rise interval (when the circuit is
    essentially in place).
    """
    steps = [r["step"] for r in history]
    acc = [r[acc_key] for r in history]
    dacc = [(acc[i + 1] - acc[i]) / max(steps[i + 1] - steps[i], 1) for i in range(len(acc) - 1)]
    ti = max(range(len(dacc)), key=lambda i: dacc[i]) if dacc else 0
    cross = lambda thr: next((steps[i] for i in range(len(acc)) if acc[i] >= thr), None)
    return {
        "transition_step": steps[ti + 1] if steps else None,
        "transition_from": steps[ti] if steps else None,
        "transition_to": steps[ti + 1] if steps else None,
        "step_at_0.5": cross(0.5),
        "step_at_0.9": cross(0.9),
        "final_acc": acc[-1] if acc else None,
    }


def load_cell_windows(cell: str, n: int = 1000, window: int = 256, seed: int = 4242):
    """(tok, aa, phase) for a 2x2 cell (A/B/C/D), via the training pool (cache or regenerate).

    A = nano_uniform (uniform codons), B = hard detector: both synthetic, regenerated on demand.
    C/D = real hg38 CDS, sampled from the genome.

    Caches are resolved against the repo's `data/cache` regardless of the working directory -- the
    training pool uses a *relative* path, so from `notebooks/` it would look in `notebooks/data/cache`,
    miss, and try to regenerate C/D from a 3 GB fasta every call (and fail outright when the genome is
    not on this machine). Here we:

      1. return the exact repo cache `cell{X}_val_{n}_{window}_{seed}.npz` if it exists;
      2. else reuse the largest repo val cache for the same cell/window/seed, sliced to n (this is how
         the notebook fills C/D from whatever a training or Phase-C run left behind);
      3. else regenerate via the pool, from the repo root so its relative cache and the fasta resolve.
         For A/B that is cheap and synthetic; for C/D it needs hg38.fa, and raises with a pointer if
         the genome is absent.
    """
    import os
    import re
    import sys
    from contextlib import contextmanager

    cell = cell.upper()

    exact = CACHE_DIR / f"cell{cell}_val_{n}_{window}_{seed}.npz"
    if exact.exists():
        z = np.load(exact)
        return z["tok"], z["aa"], z["ph"]

    pat = re.compile(rf"^cell{cell}_val_(\d+)_{window}_{seed}\.npz$")
    cached = sorted(((int(mm.group(1)), p) for p in CACHE_DIR.glob(f"cell{cell}_val_*_{window}_{seed}.npz")
                     if (mm := pat.match(p.name))), reverse=True)
    if cached:
        have_n, path = cached[0]
        z = np.load(path)
        if have_n < n:
            print(f"cell {cell}: no cache at n={n}; using {path.name} ({have_n} windows).")
        return z["tok"][:n], z["aa"][:n], z["ph"][:n]

    @contextmanager
    def _chdir(d):
        prev = os.getcwd()
        os.chdir(d)
        try:
            yield
        finally:
            os.chdir(prev)

    sys.path.insert(0, str(REPO / "scripts"))
    from train_cell import pool  # noqa: E402
    from gfmcd.data.genome import Split  # noqa: E402

    try:
        with _chdir(REPO):
            return pool(cell, n, window, Split.VAL, seed)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"cell {cell}: no val cache matches cell{cell}_val_*_{window}_{seed}.npz in {CACHE_DIR}, "
            f"and regeneration needs the genome ({e}). Copy a cached .npz here, or run on a machine "
            f"with hg38.fa + GENCODE."
        ) from None
