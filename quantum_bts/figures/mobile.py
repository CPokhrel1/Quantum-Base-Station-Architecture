import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    ax_style, BLUE, DARK, GRAY, GREEN, MOB_TYPE_COLORS, MOB_TYPE_MARKERS, PURPLE, RED,
    save_fig, TEAL,
)


def fig_mobile_fidelity_dist(mob_links):
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    metrics_info = [
        ('fidelity',   'Fidelity F(Φ⁺)',     BLUE,   (0,1)),
        ('concurrence','Concurrence C',        TEAL,   (0,1)),
        ('qber',       'QBER',                 RED,    (0,0.5)),
        ('negativity', 'Log. Negativity Eₙ',  PURPLE, (0,1)),
    ]
    for ax, (col, label, color, xlim) in zip(axes, metrics_info):
        ax_style(ax, f'({["A","B","C","D"][list(axes).index(ax)]})  {label}')
        data = mob_links[col]
        ax.hist(data, bins=25, color=color, alpha=0.82,
                edgecolor='white', lw=0.5, zorder=3)
        mu, sd = data.mean(), data.std()
        ax.axvline(mu,    color=DARK, ls='--', lw=2, label=f'μ={mu:.3f}')
        ax.axvline(mu-sd, color=GRAY, ls=':',  lw=1)
        ax.axvline(mu+sd, color=GRAY, ls=':',  lw=1)
        if col == 'fidelity':
            ax.axvline(0.5,  color=RED,   ls=':', lw=1.5, label='Ent. threshold (0.5)')
            ax.axvline(2/3,  color=GREEN, ls=':', lw=1.5, label='QKD threshold (0.667)')
        ax.set_xlabel(label, fontsize=10, color=DARK)
        ax.set_ylabel('Count', fontsize=10, color=DARK)
        ax.set_xlim(*xlim)
        ax.legend(fontsize=8)

    fig.suptitle('Mobile Node Quantum Link Metrics -- Distribution',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig03_mobile_fidelity_dist.png")


def fig_fidelity_vs_dist_speed(mob_links):
    fig, axes = plt.subplots(1,2, figsize=(16,7))
    fig.patch.set_facecolor('white')

    ax1 = axes[0]
    ax_style(ax1, '(A)  Fidelity vs Distance  (colour = type | size ∝ speed)')
    for mtype, grp in mob_links.groupby('mob_type'):
        ax1.scatter(
            grp['distance_km'], grp['fidelity'],
            c=MOB_TYPE_COLORS[mtype], marker=MOB_TYPE_MARKERS[mtype],
            s=20 + grp['speed_kmh'] * 0.85,
            alpha=0.78, edgecolors='white', lw=0.4, zorder=4,
            label=f'{mtype}')
    ax1.axhline(0.5, color=RED,   ls=':', lw=1.8, label='Entanglement threshold  (F=0.50)')
    ax1.axhline(2/3, color=GREEN, ls=':', lw=1.8, label='QKD threshold  (F=0.67)')
    for sp, lab in [(5,'5 km/h'), (40,'40 km/h'), (80,'80 km/h')]:
        ax1.scatter([], [], s=20+sp*0.85, c=GRAY, alpha=0.75,
                    marker='o', edgecolors='white', label=f'Speed ~ {lab}')
    ax1.set_xlabel('Distance to BTS (km)', fontsize=10, color=DARK)
    ax1.set_ylabel('Fidelity F(Φ⁺)',       fontsize=10, color=DARK)
    ax1.legend(fontsize=8.2, framealpha=0.95, edgecolor='#CCC',
               loc='upper right', ncol=1, title='Mobile type / speed guide',
               title_fontsize=8)
    ax1.set_xlim(left=0); ax1.set_ylim(0.2, 1.02)

    ax2 = axes[1]
    ax_style(ax2, '(B)  Fidelity vs Speed  (by Mobile Type)')
    for mtype, grp in mob_links.groupby('mob_type'):
        ax2.scatter(grp['speed_kmh'], grp['fidelity'],
                    c=MOB_TYPE_COLORS[mtype], marker=MOB_TYPE_MARKERS[mtype],
                    alpha=0.72, s=50, edgecolors='white', lw=0.4,
                    label=mtype, zorder=4)
        coef = np.polyfit(grp['speed_kmh'], grp['fidelity'], 1)
        xs = np.linspace(grp['speed_kmh'].min(), grp['speed_kmh'].max(), 80)
        ax2.plot(xs, np.polyval(coef, xs),
                 color=MOB_TYPE_COLORS[mtype], lw=2.2, ls='--', alpha=0.9)
    ax2.axhline(0.5, color=RED,   ls=':', lw=1.8, label='Entanglement threshold  (F=0.50)')
    ax2.axhline(2/3, color=GREEN, ls=':', lw=1.8, label='QKD threshold  (F=0.67)')
    ax2.plot([], [], color=GRAY, lw=2, ls='--', label='Per-type linear trend')
    ax2.set_xlabel('Speed (km/h)',    fontsize=10, color=DARK)
    ax2.set_ylabel('Fidelity F(Φ⁺)', fontsize=10, color=DARK)
    ax2.legend(fontsize=8.5, framealpha=0.95, edgecolor='#CCC',
               loc='upper right', title='Mobile type / thresholds', title_fontsize=8)
    ax2.set_ylim(0.2, 1.02)

    fig.suptitle('Mobile Node Fidelity vs Distance & Speed',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig04_fidelity_vs_dist_speed.png")


def fig_mobile_type_comparison(mob_links):
    metrics_info = [
        ('fidelity',    'Fidelity F(Φ⁺)',      BLUE,   (0, 1),    True),
        ('concurrence', 'Concurrence C',         TEAL,   (0, 1),    False),
        ('qber',        'QBER',                  RED,    (0, 0.5),  False),
        ('transmittance','Transmittance T',       GREEN,  (0, 1),    False),
    ]
    mob_types = sorted(mob_links['mob_type'].unique())
    x = np.arange(len(mob_types))
    type_colors = [MOB_TYPE_COLORS[t] for t in mob_types]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    for ax_i, (col, label, bar_col, ylim, add_thresh) in enumerate(metrics_info):
        ax = axes_flat[ax_i]
        ax_style(ax, f'({chr(65+ax_i)})  {label} by Mobile Type')

        means = [mob_links[mob_links['mob_type']==mt][col].mean() for mt in mob_types]
        stds  = [mob_links[mob_links['mob_type']==mt][col].std()  for mt in mob_types]

        bars = ax.bar(x, means, yerr=stds, capsize=6,
                      color=type_colors, alpha=0.82,
                      edgecolor=DARK, lw=0.9, width=0.55,
                      error_kw=dict(ecolor=DARK, lw=1.5, capthick=1.5))

        for b, m, s in zip(bars, means, stds):
            ax.text(b.get_x()+b.get_width()/2, m + s + ylim[1]*0.02,
                    f'{m:.3f}', ha='center', fontsize=9,
                    fontweight='bold', color=DARK)

        if add_thresh:
            ax.axhline(0.5, color=RED,   ls=':', lw=1.6,
                       label='Entanglement threshold (F=0.5)')
            ax.axhline(2/3, color=GREEN, ls=':', lw=1.6,
                       label='QKD threshold (F=0.667)')
            ax.legend(fontsize=8, framealpha=0.92, edgecolor='#CCC', loc='lower right')

        ax.set_xticks(x)
        ax.set_xticklabels(mob_types, fontsize=10, color=DARK, fontweight='bold')
        ax.set_ylabel(label, fontsize=10, color=DARK)
        ax.set_ylim(ylim[0], ylim[1] * 1.18)

        type_patches = [mpatches.Patch(color=MOB_TYPE_COLORS[t], alpha=0.82,
                                        label=t) for t in mob_types]
        if not add_thresh:
            ax.legend(handles=type_patches, fontsize=8.5, framealpha=0.92,
                      edgecolor='#CCC', loc='upper right')

    fig.suptitle('Quantum Link Metrics by Mobile Node Type  (error bars = ±1 σ)',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return save_fig(fig, "fig05_mobile_type_comparison.png")


def fig_qber_vs_speed(mob_links):
    fig, axes = plt.subplots(1,2, figsize=(14,6))
    fig.patch.set_facecolor('white')

    for ax, col, label, color in zip(
        axes,
        ['qber','concurrence'],
        ['QBER','Concurrence C'],
        [RED, TEAL]
    ):
        ax_style(ax, f'{label} vs Speed')
        for mtype, grp in mob_links.groupby('mob_type'):
            ax.scatter(grp['speed_kmh'], grp[col], alpha=0.55, s=30, label=mtype)
        coef = np.polyfit(mob_links['speed_kmh'], mob_links[col], 1)
        xs = np.linspace(0, mob_links['speed_kmh'].max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color=DARK, lw=2, ls='--', label='Linear trend')
        ax.set_xlabel('Speed (km/h)', fontsize=10, color=DARK)
        ax.set_ylabel(label, fontsize=10, color=DARK)
        ax.legend(fontsize=8.5)

    fig.suptitle('QBER & Concurrence vs Mobile Node Speed', fontsize=13,
                 fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig09_qber_vs_speed.png")
