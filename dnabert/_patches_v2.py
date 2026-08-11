"""Make the delivered notebooks survive a Run-all. Locates cells by content.

Every notebook from 06 on contains a cell that raised on the authors' last run,
followed by a cell that repairs it. They patched downwards instead of editing
upwards, which is fine when you drive a notebook by hand but fatal to Run-all:
Colab stops the queue at the first exception, so the repair below never executes.

v1 addressed cells by absolute index, which breaks the moment a cell is inserted or
removed -- and it aborted on the first miss, so later notebooks went unpatched. This
version finds each target by its own text, and treats "not found" as nothing to do
rather than an error: the text it searches for IS the failure site, so its absence
means there is nothing left to protect. Only genuine ambiguity is fatal.

Note 06: the smoke-test call appears in BOTH cell 4A and cell 4B, because 4B reruns
the same test with the corrected pooler. `forbid` distinguishes them -- 4B is the one
that defines split_and_content_mean_pool, and it must never be wrapped.

Each patch is derived from the authors' own repair cell. Nothing that computes a
result is altered; the changes only stop a superseded check from halting the run, and
every patched cell still runs to completion so later cells keep the names they need.
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
    """Wrap everything from `anchor` to end of cell in try/except."""
    idx = src.index(anchor)
    head, tail = src[:idx], src[idx:]
    return (
        head
        + f"{NOTE}{reason}\n"
        + "try:\n"
        + _indent(tail.rstrip("\n"))
        + "\n"
        + "except Exception as exc:\n"
        + '    print("[staged fix] superseded step did not run:", type(exc).__name__, exc)\n'
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

SMOKE_REASON = ("smoke test uses the 3-D pooler; cell 4B redefines it for "
                "DNABERT-2's flattened 2-D output and reruns this.")
VALIDATION_REASON = ("this key omits mutation_level; the next cell validates on the "
                     "correct key and saves.")
HIDDEN_STATES_REASON = ("the custom model returns a plain tuple; the next cell's "
                        "inspect_custom_model_output handles it.")

# prefix, kind, needle, payload, applied-marker, forbidden-text (disambiguator)
PATCHES = [
    ("06", "wrap", SMOKE_ANCHOR, SMOKE_REASON,
     NOTE + SMOKE_REASON, "def split_and_content_mean_pool"),
    ("07", "wrap", VALIDATION_ANCHOR, VALIDATION_REASON,
     NOTE + VALIDATION_REASON, None),
    ("08", "replace", PROBE_SHAPE_OLD, PROBE_SHAPE_NEW,
     NOTE + "the saved probe is a pipeline", None),
    ("08", "wrap", HIDDEN_STATES_ANCHOR, HIDDEN_STATES_REASON,
     NOTE + HIDDEN_STATES_REASON, None),
    # marker must be text only this patch inserts: the authors' own repair cell
    # already defines disallowed_unexpected_keys, which would read as "done".
    ("09", "replace", UNEXPECTED_OLD, UNEXPECTED_NEW,
     NOTE + "the checkpoint carries MLM-head weights", None),
]


def _locate(nb, needle, forbid):
    """Indices of code cells containing `needle` and not the disambiguator."""
    hits = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if needle in src and not (forbid and forbid in src):
            hits.append(i)
    return hits


def apply_patches(project_dir, log=print):
    """Apply each patch where its target still exists. Reports rather than aborts."""
    project_dir = Path(project_dir)
    by_prefix = {}
    for p in project_dir.glob("0*.ipynb"):
        by_prefix.setdefault(p.name[:2], p)

    applied = skipped = 0
    for prefix, kind, needle, payload, marker, forbid in PATCHES:
        path = by_prefix.get(prefix)
        if path is None:
            log(f"  {prefix}: notebook not found - skipped")
            skipped += 1
            continue

        nb = json.loads(path.read_text(encoding="utf-8"))

        if any(marker in "".join(c["source"]) for c in nb["cells"]):
            log(f"  {path.name:52s} already patched")
            skipped += 1
            continue

        hits = _locate(nb, needle, forbid)
        if not hits:
            # the needle IS the failure site, so its absence means nothing to protect
            log(f"  {path.name:52s} target absent (cell edited or removed) - skipped")
            skipped += 1
            continue
        assert len(hits) == 1, (
            f"{path.name}: target text found in {len(hits)} cells {hits}; "
            "refusing to guess which one to patch")

        index = hits[0]
        src = "".join(nb["cells"][index]["source"])
        patched = src.replace(needle, payload) if kind == "replace" \
            else wrap_tail(src, needle, payload)

        # a patch that breaks the cell is worse than the bug it fixes
        assert parses(patched), f"{path.name} cell {index}: patch broke the cell"
        nb["cells"][index]["source"] = patched.splitlines(keepends=True)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        applied += 1
        log(f"  {path.name:52s} cell {index:2d}  {kind}")

    log(f"\napplied {applied}, skipped {skipped}")
    return applied, skipped


def verify_patches(project_dir, log=print):
    """Count known-fatal patterns still reachable in the staged notebooks."""
    project_dir = Path(project_dir)
    by_prefix = {}
    for p in project_dir.glob("0*.ipynb"):
        by_prefix.setdefault(p.name[:2], p)

    residual = 0
    for prefix, kind, needle, payload, marker, forbid in PATCHES:
        path = by_prefix.get(prefix)
        if path is None:
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        if any(marker in "".join(c["source"]) for c in nb["cells"]):
            continue                       # patched
        if _locate(nb, needle, forbid):    # unpatched AND still present
            log(f"  RESIDUAL: {path.name} still contains the failing pattern")
            residual += 1
    return residual
