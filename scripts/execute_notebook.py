"""Execute a notebook in-process and write outputs back, without a ZMQ kernel.

The sandbox blocks the local socket a Jupyter kernel binds, so `jupyter nbconvert --execute` and
nbclient cannot run here. This executor runs every code cell in one shared namespace (exactly as a
kernel would for a top-to-bottom run), capturing stdout and any matplotlib figures into the cell's
outputs, then writes an executed copy. It is not a general kernel -- no interactivity, no `!` magics --
but it is faithful for the linear notebooks this repo generates.

    python scripts/execute_notebook.py notebooks/TEMPLATE_study_a_circuit.ipynb \
        --output study_reading_frame_counter.ipynb \
        --param CHECKPOINT=nano_uniform --param N_WINDOWS=256
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import nbformat
from nbformat.v4 import new_output


def _apply_params(src: str, params: dict[str, str]) -> str:
    """Rewrite `NAME = ...` assignments in the parameters cell. Values are eval'd (numbers/strings)."""
    for key, val in params.items():
        pattern = rf"^(\s*){re.escape(key)}\s*=.*$"
        # keep any trailing inline comment
        repl = rf"\g<1>{key} = {val!r}" if not _is_number(val) else rf"\g<1>{key} = {val}"
        src, n = re.subn(pattern, repl, src, count=1, flags=re.MULTILINE)
    return src


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def execute(nb_path: Path, out_path: Path, params: dict[str, str]) -> Path:
    nb = nbformat.read(nb_path, as_version=4)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns: dict = {"__name__": "__main__"}
    is_param_cell = True  # first code cell is the parameters cell

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if is_param_cell and params:
            src = _apply_params(src, params)
            cell.source = src
        is_param_cell = False

        cell.outputs = []
        cell.execution_count = None
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(src, "<cell>", "exec"), ns)
        except Exception:
            tb = traceback.format_exc()
            cell.outputs.append(new_output("error", ename="Error", evalue="cell failed",
                                           traceback=tb.splitlines()))
            text = buf.getvalue()
            if text:
                cell.outputs.insert(0, new_output("stream", name="stdout", text=text))
            nbformat.write(nb, out_path)
            raise RuntimeError(f"cell failed:\n{tb}")

        text = buf.getvalue()
        if text:
            cell.outputs.append(new_output("stream", name="stdout", text=text))

        # capture every open figure as a PNG display output, then close
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
            png = base64.b64encode(b.getvalue()).decode()
            cell.outputs.append(new_output("display_data", data={"image/png": png},
                                           metadata={}))
        plt.close("all")

    nbformat.write(nb, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--output", type=str, required=True, help="output filename (in same dir)")
    ap.add_argument("--param", action="append", default=[], help="NAME=VALUE, repeatable")
    args = ap.parse_args()

    params = {}
    for p in args.param:
        k, _, v = p.partition("=")
        params[k.strip()] = v.strip()

    out = args.notebook.parent / args.output
    execute(args.notebook, out, params)
    print("executed ->", out)


if __name__ == "__main__":
    main()
