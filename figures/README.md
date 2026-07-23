# Figures

> **Tier: T1 — canonical.** Current, authoritative. Read this.

One folder per study: `figures/<study>/`.

## Rules

1. **A figure must declare its source artifact(s).** `save_figure(...)` refuses to write
   without them, and emits `<name>.provenance.json` containing the git commit and the
   **sha256 of every source file**. A figure that cannot name and hash its data is
   decoration, not evidence — it will drift from the numbers the moment a run is re-done.
2. **Figure scripts do not compute science.** They read artifacts produced by
   `scripts/<study>/run_*.py`. If a number appears in a figure and nowhere in a manifest-
   stamped artifact, that is a bug.
3. **Reusable primitives live in `src/gfmcd/figures/`**, not here. Each primitive encodes a
   *shape of argument*: `grouped_bars_ci` (effect present, collapses under intervention),
   `ranked_dots` (one component dominates a scan), `interaction_plot` (a 2x2 factorial —
   non-parallel lines are the finding).
4. **A study's style lives in one module inside its folder**, e.g.
   `splicebert/figstyle.py`: fonts, palette tokens, DPI, and the PDF+PNG save helper. Every
   `make_fig*.py` of that study imports it and defines no style of its own, so a colour cannot
   mean one thing in one panel and something else in the next. The module composes the
   primitives above rather than duplicating them, and re-exports validated colour values from
   wherever they were validated rather than copying them.

## Studies

- `splicebert/` — same-intron boundary coupling and its causal localisation to L3H1.
