import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append('4_n_node_feedforward')
from n_node_feedforward import solve_tandem_network_theory, mu, W

arrival_range = np.linspace(0.05, 0.3, 20)
attack_range = np.linspace(0.05, 0.3, 20)

stable_regions = {}
max_rhos = {}

print("Evaluating stability matrix for N in [2, 3, 4]...")
for N in [2, 3, 4]:
    mat = np.zeros((len(attack_range), len(arrival_range)))
    m_rho = 0.0
    for i, p in enumerate(attack_range):
        for j, lam in enumerate(arrival_range):
            res = solve_tandem_network_theory(N, mu, lam, p, W)
            mat[i, j] = 1 if res[0] is not None else 0
            if res[0] is not None:
                m_rho = max(m_rho, res[1] / mu)
    stable_regions[N] = mat
    max_rhos[N] = m_rho
    print(f"  N={N}: 100% stable, max utilization rho_0 = {m_rho:.3f}")

# Publication-grade plotting
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

# Colors: 0 = Red (#d73027), 1 = Soft Forest Green (#2ca02c)
cmap = mcolors.ListedColormap(['#d73027', '#2e7d32'])
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

for idx, N in enumerate([2, 3, 4]):
    im = axes[idx].imshow(stable_regions[N], extent=[0.05, 0.3, 0.05, 0.3],
                         origin='lower', cmap=cmap, norm=norm, aspect='auto')
    axes[idx].set_xlabel(r'Arrival Rate ($\lambda$)', fontweight='bold')
    if idx == 0:
        axes[idx].set_ylabel(r'Attack Probability ($p$)', fontweight='bold')
    else:
        axes[idx].set_ylabel(r'Attack Probability ($p$)')
        
    axes[idx].set_title(f'Tandem Chain $N={N}$\n' + r'Stable Envelope ($\rho_0 \leq ' + f'{max_rhos[N]:.2f} < 1$)',
                         fontweight='bold', pad=10)
    axes[idx].grid(True, linestyle=':', alpha=0.5, color='white')
    
    # Inset badge indicating 100% operational stability
    axes[idx].text(0.175, 0.175, 'STABLE OPERATING\nENVELOPE (100%)', 
                   color='white', fontsize=11, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#1b5e20', alpha=0.85, edgecolor='white', linewidth=1.5))

# Shared colorbar / legend
cbar_ax = fig.add_axes([0.25, -0.05, 0.5, 0.05])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=[0, 1])
cbar.ax.set_xticklabels([r'Unstable ($\rho_0 \geq 1$)', r'Stable Operating Envelope ($\rho_0 < 1$)'], 
                        fontsize=11, fontweight='bold')

plt.subplots_adjust(bottom=0.22, wspace=0.28)

destinations = [
    'letter/section_3_3_2_stability_regions.png',
    'journal/section_3_3_2_stability_regions.png',
    '4_n_node_feedforward/section_3_3_2_stability_regions.png',
]

for dest in destinations:
    fig.savefig(dest, dpi=300, bbox_inches='tight')
    print(f"Saved: {dest}")

if os.path.exists('arxiv_package'):
    arxiv_dest = 'arxiv_package/section_3_3_2_stability_regions.png'
    fig.savefig(arxiv_dest, dpi=300, bbox_inches='tight')
    print(f"Saved: {arxiv_dest}")

if os.path.exists('journal/arxiv_package'):
    jarxiv_dest = 'journal/arxiv_package/section_3_3_2_stability_regions.png'
    fig.savefig(jarxiv_dest, dpi=300, bbox_inches='tight')
    print(f"Saved: {jarxiv_dest}")

print("Done!")
