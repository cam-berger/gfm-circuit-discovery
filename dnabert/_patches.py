"""Make the delivered notebooks survive a Run-all.

Every notebook from 06 on contains a cell that raised on the authors' last run,
followed by a cell that repairs it. They patched downwards instead of editing
upwards, which is fine when you drive a notebook by hand but fatal to Run-all:
Colab stops the queue at the first exception, so the repair below never executes.

Each patch here is derived from the authors' own repair cell -- the allow-list, the
corrected key, the shape-tolerant pooler are all theirs, not invented. Nothing that
computes a result is altered; the changes only stop a superseded check from halting
the run. Where a failing cell still defines state the notebook needs, the fix keeps
that cell running rather than skipping it.

  06 cell 11  smoke test calls content_mean_pool, which indexes [batch, tok, hid] but
              DNABERT-2 returns flattened unpadded 2-D activations. Cell 12 redefines
              the pooler, extract_layer5_representations and score_sequences_with_probe,
              then reruns this exact smoke test -- and cell 14 (the real scan) calls the
              redefined version. Only the smoke test is neutralised; every model,
              tokenizer and probe binding earlier in cell 11 still runs.
  07 cell 6   asserts uniqueness of (source_group, k), but each source appears at
              mutation level 4 and 8, so duplicates are expected. Cell 7 states this and
              re-validates on (source_group, mutation_level, k), then saves. Nothing is
              assigned after the assert, so deferring the whole block to cell 7 is safe.
  08 cell 8   prints getattr(layer5_probe, "coef_", None).shape. The saved probe is a
              pipeline, not a bare linear model, so coef_ is legitimately absent and the
              print crashes -- 149 lines before backbone, layer1, tokenize_sequences and
              run_backbone are bound. Diagnostic only; made tolerant.
  08 cell 9   prints outputs.hidden_states, but the custom model returns a plain tuple.
              Cell 10's inspect_custom_model_output exists precisely for this. Nothing
              is assigned after it.
  09 cell 6   raises on ANY unexpected checkpoint key, but the DNABERT-2 checkpoint
              carries MLM-head weights an encoder-only BertModel cannot use. Cell 7
              allows exactly "cls.predictions."; that allow-list is applied here so the
              cell finishes and still binds model, backbone and the layer modules.
"""

import ast
import json
import re
from pathlib import Path

NOTE = "# [staged fix] "
MAGIC = re.compile(r"^\s*[!%]")


def parses(src):
    """Parse-check tolerating IPython magics, which are not Python."""
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        pass
    cleaned, skipping = [], False
    for line in src.split("\n"):
        if skipping or MAGIC.match(line):
            skipping = line.rstrip().endswith("\\")
            cleaned.append("pass")
        else:
            cleaned.append(line)
    try:
        ast.parse("\n".join(cleaned))
        return True
    except SyntaxError:
        return False


def _indent(text, pad="    "):
    return "\n".join(pad + ln if ln.strip() else ln for ln in text.split("\n"))


def wrap_tail(src, anchor, reason):
    """Wrap everything from `anchor` to end of cell in try/except.

    Used where the failing statement is superseded by a later cell and nothing after
    it in this cell binds a name the notebook needs.
    """
    idx = src.index(anchor)
    head, tail = src[:idx], src[idx:]
    return (
        head
        + f"{NOTE}{reason}\n"
        + "try:\n"
        + _indent(tail.rstrip("\n"))
        + "\n"
        + "except Exception as exc:\n"
        + f'    print("[staged fix] superseded step did not run:", type(exc).__name__, exc)\n'
        + '    print("[staged fix] the following cell redoes this correctly.")\n'
    )


PROBE_SHAPE_OLD = '''print(
    "Probe coefficient shape:",
    getattr(layer5_probe, "coef_", None).shape,
)'''

PROBE_SHAPE_NEW = '''# [staged fix] the saved probe is a pipeline, so coef_ is legitimately absent.
# The next cell's find_linear_estimator() unwraps it properly.
_probe_coefficients = getattr(layer5_probe, "coef_", None)

print(
    "Probe coefficient shape:",
    "not a bare linear estimator"
    if _probe_coefficients is None
    else _probe_coefficients.shape,
)'''

UNEXPECTED_OLD = '''if unexpected_keys:
    raise RuntimeError(
        "Unexpected checkpoint parameters remain:\\n"
        + "\\n".join(
            unexpected_keys[:50]
        )
    )'''

UNEXPECTED_NEW = '''# [staged fix] the checkpoint carries MLM-head weights this encoder-only
# BertModel cannot use. Allow exactly what the next cell allows.
disallowed_unexpected_keys = [
    key
    for key in unexpected_keys
    if not key.startswith("cls.predictions.")
]

if disallowed_unexpected_keys:
    raise RuntimeError(
        "Unexpected non-MLM parameters remain:\\n"
        + "\\n".join(
            disallowed_unexpected_keys[:50]
        )
    )'''

SMOKE_ANCHOR = '''(
    smoke_representations,
    smoke_logits,
    smoke_probabilities,
) = score_sequences_with_probe('''

VALIDATION_ANCHOR = '''assert (
    heldout_reconstruction_df[
        ["source_group", "k"]
    ]'''

HIDDEN_STATES_ANCHOR = '''print(
    "Number of hidden-state tensors:",
    len(
        outputs.hidden_states
    ),
)'''

# (notebook prefix, cell index, kind, payload)
PATCHES = [
    ("06", 11, "wrap", (SMOKE_ANCHOR,
                        "smoke test uses the 3-D pooler; cell 12 redefines it for "
                        "DNABERT-2's flattened 2-D output and reruns this.")),
    ("07", 6, "wrap", (VALIDATION_ANCHOR,
                       "this key omits mutation_level; the next cell validates on the "
                       "correct key and saves.")),
    ("08", 8, "replace", (PROBE_SHAPE_OLD, PROBE_SHAPE_NEW)),
    ("08", 9, "wrap", (HIDDEN_STATES_ANCHOR,
                       "the custom model returns a plain tuple; the next cell's "
                       "inspect_custom_model_output handles it.")),
    ("09", 6, "replace", (UNEXPECTED_OLD, UNEXPECTED_NEW)),
]


def apply_patches(project_dir, log=print):
    """Apply every patch, verifying each lands exactly once and still parses."""
    project_dir = Path(project_dir)
    by_prefix = {p.name[:2]: p for p in project_dir.glob("0*.ipynb")}
    applied = 0

    for prefix, cell_index, kind, payload in PATCHES:
        path = by_prefix.get(prefix)
        assert path is not None, f"notebook {prefix} not staged"
        nb = json.loads(path.read_text(encoding="utf-8"))

        cells = nb["cells"]
        assert cell_index < len(cells), f"{path.name}: no cell {cell_index}"
        cell = cells[cell_index]
        assert cell["cell_type"] == "code", f"{path.name} cell {cell_index} is not code"
        src = "".join(cell["source"])

        if kind == "replace":
            old, new = payload
            count = src.count(old)
            assert count == 1, f"{path.name} cell {cell_index}: pattern matched {count}x"
            patched = src.replace(old, new)
        else:
            anchor, reason = payload
            count = src.count(anchor)
            assert count == 1, f"{path.name} cell {cell_index}: anchor matched {count}x"
            patched = wrap_tail(src, anchor, reason)

        # a patch that breaks the cell is worse than the bug it fixes
        assert parses(patched), f"{path.name} cell {cell_index}: patch broke the cell"
        cell["source"] = patched.splitlines(keepends=True)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        applied += 1
        log(f"  {path.name:52s} cell {cell_index:2d}  {kind}")

    assert applied == len(PATCHES), "not every patch applied"
    return applied


def verify_patches(project_dir, log=print):
    """Confirm no known-fatal pattern survives in the staged notebooks."""
    project_dir = Path(project_dir)
    by_prefix = {p.name[:2]: p for p in project_dir.glob("0*.ipynb")}
    residual = 0

    for prefix, cell_index, kind, payload in PATCHES:
        src = "".join(
            json.loads(by_prefix[prefix].read_text(encoding="utf-8"))
            ["cells"][cell_index]["source"]
        )
        if kind == "replace":
            old, _ = payload
            if old in src:
                log(f"  RESIDUAL: {by_prefix[prefix].name} cell {cell_index}")
                residual += 1
        elif NOTE not in src:
            log(f"  RESIDUAL: {by_prefix[prefix].name} cell {cell_index} not wrapped")
            residual += 1
    return residual
