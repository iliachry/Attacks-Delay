"""
Generate publication-grade vector (.pdf) and raster (.png) figures for the
Saddle-Node Retry Bifurcation & Dual-Branch Phase Portrait in aqnet v0.2.0.
Enforces TrueType (Type 42) font embedding (0 Type-3 fonts).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aqnet.models.bifurcation import (
    solve_dual_fixed_points,
    solve_saddle_node_bifurcation,
    throughput_curve_gamma,
)


def generate_saddle_node_figure() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.8,
        }
    )

    mu = 10.0
    T = 1.0
    p = 0.25
    sn_mod = solve_saddle_node_bifurcation("one_node_modification", p=p, mu=mu, T=T)
    lam_slice = 3.50
    dual_slice = solve_dual_fixed_points(lam_slice, "one_node_modification", p=p, mu=mu, T=T)

    lam_grid_stable = np.linspace(0.0, sn_mod.Lambda_sn, 250)
    lam_grid_unstable = np.linspace(sn_mod.Lambda_sn, mu - 1e-4, 250)
    gamma_stable = [
        throughput_curve_gamma(float(x), "one_node_modification", p=p, mu=mu, T=T)
        for x in lam_grid_stable
    ]
    gamma_unstable = [
        throughput_curve_gamma(float(x), "one_node_modification", p=p, mu=mu, T=T)
        for x in lam_grid_unstable
    ]

    # Local quadratic normal form around Lambda_sn
    lam_nf = np.linspace(sn_mod.Lambda_sn - 2.4, min(mu, sn_mod.Lambda_sn + 2.4), 160)
    gamma_nf = sn_mod.lambda_crit_sn - sn_mod.c2_normal_form * ((lam_nf - sn_mod.Lambda_sn) ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=300)

    # Panel (a): Unimodal Throughput Fold & Dual Fixed-Point Branches
    ax1.plot(
        lam_grid_stable,
        gamma_stable,
        color="#1d4ed8",
        linewidth=2.4,
        label=r"Stable lower branch $\Lambda_{\mathrm{stable}}^*(\lambda)$ ($\Gamma' > 0$)",
    )
    ax1.plot(
        lam_grid_unstable,
        gamma_unstable,
        color="#b91c1c",
        linestyle="--",
        linewidth=2.2,
        label=r"Unstable separatrix $\Lambda_{\mathrm{unstable}}^*(\lambda)$ ($\Gamma' < 0$)",
    )
    ax1.plot(
        lam_nf,
        gamma_nf,
        color="#64748b",
        linestyle=":",
        linewidth=1.6,
        label=r"Quadratic normal form $\lambda_{\mathrm{crit}}^{\mathrm{SN}} - c_2(\Lambda^* - \Lambda_{\mathrm{SN}}^*)^2$",
    )
    ax1.axhline(
        lam_slice,
        color="#0f766e",
        linestyle="-.",
        linewidth=1.3,
        label=rf"Subcritical arrival $\lambda = {lam_slice:.1f}$ pkt/s",
    )
    ax1.scatter(
        [sn_mod.Lambda_sn],
        [sn_mod.lambda_crit_sn],
        color="#d97706",
        edgecolor="#1e293b",
        s=75,
        zorder=6,
        label=rf"Saddle-Node fold $(\Lambda_{{\mathrm{{SN}}}}^*={sn_mod.Lambda_sn:.2f},\,\lambda_{{\mathrm{{crit}}}}^{{\mathrm{{SN}}}}={sn_mod.lambda_crit_sn:.2f})$",
    )
    ax1.scatter(
        [dual_slice.Lambda_stable, dual_slice.Lambda_unstable],
        [lam_slice, lam_slice],
        color=["#1d4ed8", "#b91c1c"],
        s=50,
        zorder=6,
    )
    ax1.annotate(
        rf"Separatrix gap $\Delta\Lambda = {dual_slice.separatrix_gap:.2f}$",
        xy=((dual_slice.Lambda_stable + dual_slice.Lambda_unstable) / 2.0, lam_slice),
        xytext=(0.45, 4.85),
        arrowprops={"arrowstyle": "->", "color": "#0f766e", "lw": 1.1},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
        fontsize=9,
        color="#0f766e",
        fontweight="bold",
    )
    ax1.set_xlim(0.0, mu)
    ax1.set_ylim(0.0, 6.2)
    ax1.set_xlabel(r"Total Internal Attempt Rate $\Lambda^*$ (pkt/s)")
    ax1.set_ylabel(r"Conserved External Throughput $\Gamma(\Lambda^*) = \Lambda^* P_{\mathrm{succ}}(\Lambda^*)$")
    ax1.set_title(r"(a) Saddle-Node Fold & Dual Branches ($\mu=10, T=1, p=0.25$)")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="lower left", framealpha=0.92)

    # Panel (b): Critical External Capacity vs Attack Probability p
    p_grid = np.linspace(0.0, 0.92, 100)
    crit_dest = [
        solve_saddle_node_bifurcation("one_node_destruction", p=float(pv), mu=mu, T=T).lambda_crit_sn
        for pv in p_grid
    ]
    crit_mod = [
        solve_saddle_node_bifurcation("one_node_modification", p=float(pv), mu=mu, T=T).lambda_crit_sn
        for pv in p_grid
    ]
    naive_bound = [mu * (1.0 - float(pv)) for pv in p_grid]

    ax2.plot(
        p_grid,
        naive_bound,
        color="#475569",
        linestyle=":",
        linewidth=1.8,
        label=r"Naive non-timeout bound $\mu(1-p)$",
    )
    ax2.plot(
        p_grid,
        crit_dest,
        color="#059669",
        linewidth=2.3,
        label=rf"Case 1 (Destruction) $\lambda_{{\mathrm{{crit}}}}^{{\mathrm{{SN,dest}}}}(p) \equiv {crit_dest[0]:.2f}$ ($p$-invariant)",
    )
    ax2.plot(
        p_grid,
        crit_mod,
        color="#1d4ed8",
        linewidth=2.3,
        label=r"Case 2 (Modification) $\lambda_{\mathrm{crit}}^{\mathrm{SN,mod}}(p) = (1-p)\lambda_{\mathrm{crit}}^{\mathrm{SN}}(0)$",
    )
    ax2.fill_between(
        p_grid,
        crit_mod,
        naive_bound,
        color="#fca5a5",
        alpha=0.28,
        label="Timeout-induced collapse gap (Case 2)",
    )
    ax2.set_xlim(0.0, 0.90)
    ax2.set_ylim(0.0, 10.5)
    ax2.set_xlabel(r"Adversarial Attack Probability $p$")
    ax2.set_ylabel(r"Critical External Arrival Rate $\lambda_{\mathrm{crit}}^{\mathrm{SN}}$ (pkt/s)")
    ax2.set_title(r"(b) Critical Capacity Manifold: Destruction vs. Modification")
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()

    repo_root = Path(__file__).resolve().parent.parent
    target_dirs = [
        repo_root,
        repo_root / "letter",
        repo_root / "journal",
        repo_root / "arxiv_package",
        repo_root / "journal" / "arxiv_package",
    ]
    for d in target_dirs:
        if d.exists():
            pdf_path = d / "fig_saddle_node_bifurcation.pdf"
            png_path = d / "fig_saddle_node_bifurcation.png"
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {pdf_path} & {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    generate_saddle_node_figure()
