"""Reusable figure library.

Study-specific figure SCRIPTS live in ``figures/<study>/``; the reusable primitives
live here so panels look the same across studies and carry provenance by default.
"""
from gfmcd.figures.io import figure_provenance, save_figure
from gfmcd.figures.panels import grouped_bars_ci, interaction_plot, ranked_dots
from gfmcd.figures.style import COLORS, panel_label, use_style

__all__ = ["COLORS", "figure_provenance", "grouped_bars_ci", "interaction_plot",
           "panel_label", "ranked_dots", "save_figure", "use_style"]
