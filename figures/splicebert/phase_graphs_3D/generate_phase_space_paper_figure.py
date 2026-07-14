"""Generate a publication-quality dual-carrier phase-space figure."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("/workspace")
T_LONG = 10.62994
A_CODON = 1.0
A_LONG = 0.578


def wrap_phase(theta: np.ndarray | float) -> np.ndarray:
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def joint_weight(theta3: np.ndarray, theta10: np.ndarray) -> np.ndarray:
    return np.exp(A_CODON * np.cos(theta3) + A_LONG * np.cos(theta10))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def draw_phase_surface(ax: plt.Axes, elev: float, azim: float, compact: bool = False):
    theta3 = np.linspace(-np.pi, np.pi, 181)
    theta10 = np.linspace(-np.pi, np.pi, 181)
    xx, yy = np.meshgrid(theta3, theta10)
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
        alpha=0.84,
        shade=True,
    )

    n = np.arange(0, 121)
    phase3 = wrap_phase(2 * np.pi * n / 3)
    phase10 = wrap_phase(2 * np.pi * n / T_LONG)
    point_z = joint_weight(phase3, phase10) + 0.22
    lane_styles = [
        ("#D55E00", "o", r"codon lane 0: $n\,\mathrm{mod}\,3=0$"),
        ("#0072B2", "s", r"codon lane 1: $n\,\mathrm{mod}\,3=1$"),
        ("#009E73", "D", r"codon lane 2: $n\,\mathrm{mod}\,3=2$"),
    ]
    lane_phases = [0.0, 2 * np.pi / 3, -2 * np.pi / 3]

    rail_phase10 = np.linspace(-np.pi, np.pi, 500)
    for lane_phase, (color, _, _) in zip(lane_phases, lane_styles):
        rail_phase3 = np.full_like(rail_phase10, lane_phase)
        rail_z = joint_weight(rail_phase3, rail_phase10) + 0.19
        ax.plot(
            rail_phase3,
            rail_phase10,
            rail_z,
            color="white",
            linewidth=3.6 if not compact else 2.2,
            alpha=0.9,
            solid_capstyle="round",
            zorder=23,
        )
        ax.plot(
            rail_phase3,
            rail_phase10,
            rail_z + 0.01,
            color=color,
            linewidth=2.0 if not compact else 1.25,
            solid_capstyle="round",
            zorder=24,
        )

    scatters = []
    for lane, (color, marker, label) in enumerate(lane_styles):
        mask = n % 3 == lane
        scatter = ax.scatter(
            phase3[mask],
            phase10[mask],
            point_z[mask],
            s=30 if not compact else 14,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.85 if not compact else 0.45,
            depthshade=False,
            label=label,
            zorder=25,
        )
        scatters.append(scatter)

    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.08, 1.08, 0.62))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(0, 5.15)
    ax.set_xticks([-np.pi, -2 * np.pi / 3, 0, 2 * np.pi / 3, np.pi])
    ax.set_xticklabels(
        [r"$-\pi$", r"$-2\pi/3$", "0", r"$2\pi/3$", r"$\pi$"]
    )
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_zticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel(r"Codon phase, $\theta_3$", labelpad=9)
    ax.set_ylabel(r"10.63-nt phase, $\theta_{10}$", labelpad=9)
    ax.set_zlabel("")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 0.55))
        axis.pane.set_edgecolor((0.72, 0.72, 0.72, 0.75))
    ax.grid(True, alpha=0.2, linewidth=0.45)

    return surface, scatters


def make_candidate_grid() -> None:
    angles = [(28, -55), (32, -45), (24, -65), (36, -55)]
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    for index, (elev, azim) in enumerate(angles, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        draw_phase_surface(ax, elev=elev, azim=azim, compact=True)
        ax.set_title(f"elevation {elev}°, azimuth {azim}°", pad=2)
    fig.savefig(OUT / "phase-space-angle-candidates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_final_figure(elev: float = 31, azim: float = -55) -> None:
    fig = plt.figure(figsize=(8.2, 6.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    surface, scatters = draw_phase_surface(ax, elev=elev, azim=azim)

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
        handles=scatters,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        framealpha=0.94,
        edgecolor="0.8",
    )

    save_options = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    fig.savefig(OUT / "carrier-phase-space-paper.png", dpi=450, **save_options)
    fig.savefig(OUT / "carrier-phase-space-paper.pdf", **save_options)
    fig.savefig(OUT / "carrier-phase-space-paper.svg", **save_options)
    plt.close(fig)


if __name__ == "__main__":
    configure_style()
    make_candidate_grid()
    make_final_figure()
