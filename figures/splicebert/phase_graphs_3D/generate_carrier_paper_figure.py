"""Generate publication-quality views of the period-3/10.63 carrier interaction."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("/workspace")
T_LONG = 10.62994
A_CODON = 1.0
A_LONG = 0.578


def joint_weight(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    logit = (
        A_CODON * np.cos(2 * np.pi * x / 3)
        + A_LONG * np.cos(2 * np.pi * y / T_LONG)
    )
    return np.exp(logit)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def draw_surface(ax: plt.Axes, elev: float, azim: float, compact: bool = False):
    x = np.linspace(-15, 15, 181)
    y = np.linspace(-15, 15, 181)
    xx, yy = np.meshgrid(x, y)
    zz = joint_weight(xx, yy)

    norm = mpl.colors.Normalize(vmin=float(zz.min()), vmax=float(zz.max()))
    surface = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap="viridis",
        norm=norm,
        rcount=121,
        ccount=121,
        linewidth=0,
        antialiased=True,
        alpha=0.91,
        shade=True,
    )

    t = np.linspace(-15, 15, 900)
    diagonal = joint_weight(t, t)
    ax.plot(
        t,
        t,
        diagonal + 0.12,
        color="white",
        linewidth=4.2 if not compact else 3.2,
        alpha=0.9,
        solid_capstyle="round",
        zorder=20,
    )
    trajectory, = ax.plot(
        t,
        t,
        diagonal + 0.14,
        color="#D55E00",
        linewidth=2.5 if not compact else 1.8,
        label=r"physical trajectory $x=y$",
        solid_capstyle="round",
        zorder=21,
    )

    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.12, 1.12, 0.58))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(0, 5.15)
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_zticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel(r"Period-3 coordinate, $x$ (nt)", labelpad=8)
    ax.set_ylabel(r"10.63-nt coordinate, $y$ (nt)", labelpad=9)
    ax.set_zlabel("")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 0.55))
        axis.pane.set_edgecolor((0.72, 0.72, 0.72, 0.75))
    ax.grid(True, alpha=0.2, linewidth=0.45)

    return surface, trajectory


def make_candidate_grid() -> None:
    angles = [(28, -55), (32, -45), (24, -65), (36, -55)]
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    for index, (elev, azim) in enumerate(angles, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        draw_surface(ax, elev=elev, azim=azim, compact=True)
        ax.set_title(f"elevation {elev}°, azimuth {azim}°", pad=2)
    fig.savefig(OUT / "carrier-angle-candidates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_final_figure(elev: float = 31, azim: float = -55) -> None:
    fig = plt.figure(figsize=(8.2, 6.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface, trajectory = draw_surface(ax, elev=elev, azim=azim)

    colorbar = fig.colorbar(
        surface,
        ax=ax,
        shrink=0.64,
        pad=0.035,
        aspect=24,
    )
    colorbar.set_label(
        r"Unnormalized attention weight $e^{C_3+C_{10}}$",
        rotation=90,
        labelpad=10,
    )
    ax.legend(
        handles=[trajectory],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        frameon=True,
        framealpha=0.94,
        edgecolor="0.8",
    )

    fig.savefig(OUT / "carrier-wave-interaction-paper.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / "carrier-wave-interaction-paper.pdf", bbox_inches="tight")
    fig.savefig(OUT / "carrier-wave-interaction-paper.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    configure_style()
    make_candidate_grid()
    make_final_figure()
