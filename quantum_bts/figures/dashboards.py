import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    ax_style, BLUE, DARK, GRAY, GREEN, MOB_TYPE_COLORS, MOB_TYPE_MARKERS, ORANGE,
    PURPLE, RED, save_fig, TEAL,
)


def fig_dashboard_bts(G):
    edges = list(G.edges(data=True))
    if not edges: print("  [!] No BTS edges for dashboard."); return None
    metrics_info = [
        ('fidelity',    'Fidelity F(Φ⁺)',    BLUE,   (0, 1),   0.5,  2/3),
        ('concurrence', 'Concurrence C',       TEAL,   (0, 1),   0.0,  None),
        ('qber',        'QBER',                RED,    (0, 0.5), None, None),
        ('negativity',  'Log. Negativity Eₙ', PURPLE, (0, 1),   None, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()
    for ax_i, (col, label, color, xlim, thr1, thr2) in enumerate(metrics_info):
        ax = axes_flat[ax_i]
        ax_style(ax, f'({chr(65+ax_i)})  BTS Link -- {label}')
        vals = np.array([e[2][col] for e in edges])
        ax.hist(vals, bins=14, color=color, alpha=0.82, edgecolor='white',
                lw=0.5, zorder=3)
        mu, sd = vals.mean(), vals.std()
        ax.axvline(mu,    color=DARK, ls='--', lw=2.0, label=f'μ = {mu:.3f}')
        ax.axvline(mu-sd, color=GRAY, ls=':',  lw=1.2, label=f'μ±σ  (σ={sd:.3f})')
        ax.axvline(mu+sd, color=GRAY, ls=':',  lw=1.2)
        if thr1: ax.axvline(thr1, color=RED,   ls=':', lw=1.6,
                            label=f'Ent. threshold ({thr1:.2f})')
        if thr2: ax.axvline(thr2, color=GREEN, ls=':', lw=1.6,
                            label=f'QKD threshold ({thr2:.3f})')
        ax.set_xlabel(label, fontsize=10, color=DARK)
        ax.set_ylabel('Count', fontsize=10, color=DARK)
        ax.set_xlim(*xlim)
        ax.legend(fontsize=8, framealpha=0.92, edgecolor='#CCC')
    fig.suptitle('BTS Backbone Quantum Link Metrics -- Distribution Dashboard',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return save_fig(fig, "fig13a_dashboard_bts.png")


def fig_dashboard_mobile(mob_links):
    metrics_info = [
        ('fidelity',    'Fidelity F(Φ⁺)',    BLUE,   (0, 1),   0.5,  2/3),
        ('concurrence', 'Concurrence C',       TEAL,   (0, 1),   None, None),
        ('qber',        'QBER',                RED,    (0, 0.5), 0.11, 0.25),
        ('negativity',  'Log. Negativity Eₙ', PURPLE, (0, 1),   None, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()
    for ax_i, (col, label, color, xlim, thr1, thr2) in enumerate(metrics_info):
        ax = axes_flat[ax_i]
        ax_style(ax, f'({chr(65+ax_i)})  Mobile Link -- {label}')
        vals = mob_links[col].values
        ax.hist(vals, bins=20, color=color, alpha=0.65, edgecolor='white',
                hatch='////', lw=0.4, zorder=3)
        for mtype, grp in mob_links.groupby('mob_type'):
            ax.hist(grp[col].values, bins=16, alpha=0.35,
                    color=MOB_TYPE_COLORS[mtype], edgecolor='none',
                    label=mtype, zorder=4, histtype='stepfilled')
        mu, sd = vals.mean(), vals.std()
        ax.axvline(mu,    color=DARK, ls='--', lw=2.0, label=f'μ = {mu:.3f}')
        ax.axvline(mu-sd, color=GRAY, ls=':',  lw=1.2, label=f'σ = {sd:.3f}')
        ax.axvline(mu+sd, color=GRAY, ls=':',  lw=1.2)
        if thr1: ax.axvline(thr1, color=RED,   ls=':', lw=1.6,
                            label=f'Threshold ({thr1})')
        if thr2: ax.axvline(thr2, color=GREEN, ls=':', lw=1.6,
                            label=f'Threshold ({thr2})')
        ax.set_xlabel(label, fontsize=10, color=DARK)
        ax.set_ylabel('Count', fontsize=10, color=DARK)
        ax.set_xlim(*xlim)
        ax.legend(fontsize=7.5, framealpha=0.92, edgecolor='#CCC', ncol=2)
    fig.suptitle('Mobile Node Quantum Link Metrics -- Distribution Dashboard'
                 '\n(hatch = all mobile | shaded = per type)',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, "fig13b_dashboard_mobile.png")


def fig_dashboard_comparison(G, mob_links):
    edges = list(G.edges(data=True))
    if not edges: return None
    bts_fids  = np.array([e[2]['fidelity']    for e in edges])
    bts_dists = np.array([e[2]['distance_km'] for e in edges])
    mob_fids  = mob_links['fidelity'].values
    mob_dists = mob_links['distance_km'].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('white')

    ax1 = axes[0]
    ax_style(ax1, '(A)  Fidelity vs Distance -- BTS & Mobile')
    ax1.scatter(bts_dists, bts_fids, c=BLUE, s=90, alpha=0.82,
                edgecolors=DARK, lw=0.8, marker='o', zorder=5,
                label='BTS backbone links')
    for mtype, grp in mob_links.groupby('mob_type'):
        ax1.scatter(grp['distance_km'], grp['fidelity'],
                    c=MOB_TYPE_COLORS[mtype], s=22, alpha=0.55,
                    marker=MOB_TYPE_MARKERS[mtype], edgecolors='none',
                    zorder=4, label=f'Mobile -- {mtype}')
    ax1.axhline(0.5, color=RED,   ls=':', lw=1.8,
                label='Entanglement threshold (F=0.5)')
    ax1.axhline(2/3, color=GREEN, ls=':', lw=1.8,
                label='QKD threshold (F=0.667)')
    ax1.set_xlabel('Distance (km)', fontsize=10, color=DARK)
    ax1.set_ylabel('Fidelity F(Φ⁺)', fontsize=10, color=DARK)
    ax1.legend(fontsize=7.8, framealpha=0.95, edgecolor='#CCC',
               loc='upper right', ncol=1)

    ax2 = axes[1]
    ax_style(ax2, '(B)  Fidelity Distribution by Link Category')
    box_data, box_labels, box_colors = [bts_fids], ['BTS\nbackbone'], [BLUE]
    for mtype in ['pedestrian', 'vehicle', 'IoT']:
        sub = mob_links[mob_links['mob_type'] == mtype]['fidelity'].values
        if len(sub):
            box_data.append(sub)
            box_labels.append(f'Mobile\n{mtype}')
            box_colors.append(MOB_TYPE_COLORS[mtype])
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                     medianprops=dict(color=DARK, lw=2.5),
                     whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5),
                     flierprops=dict(marker='o', ms=3, alpha=0.4))
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.72)
    ax2.axhline(0.5, color=RED,   ls=':', lw=1.6,
                label='Entanglement threshold (F=0.5)')
    ax2.axhline(2/3, color=GREEN, ls=':', lw=1.6,
                label='QKD threshold (F=0.667)')
    ax2.set_ylabel('Fidelity F(Φ⁺)', fontsize=10, color=DARK)
    ax2.legend(fontsize=8.5, framealpha=0.95, edgecolor='#CCC',
               loc='lower right')
    ax2.tick_params(axis='x', labelsize=9)

    ax3 = axes[2]
    ax3.set_facecolor('white')
    best_fid = mob_links.groupby('mobile_id')['fidelity'].max()
    n_qkd  = int((best_fid > 2/3).sum())
    n_ent  = int(((best_fid > 0.5) & (best_fid <= 2/3)).sum())
    n_sub  = int((best_fid <= 0.5).sum())
    labels_pie = [f'QKD-ready\n(F>0.667)\n{n_qkd}',
                  f'Entangled\n(0.5<F<=0.667)\n{n_ent}',
                  f'Below threshold\n(F<=0.5)\n{n_sub}']
    pie_colors = [GREEN, ORANGE, RED]
    wedges, texts, autotexts = ax3.pie(
        [n_qkd, n_ent, n_sub], labels=labels_pie,
        colors=pie_colors, autopct='%1.1f%%',
        wedgeprops=dict(edgecolor='white', lw=2.5),
        textprops=dict(fontsize=8.5, color=DARK),
        startangle=90)
    for at in autotexts: at.set_fontsize(9); at.set_fontweight('bold')
    ax3.set_title('(C)  Mobile Coverage Breakdown\n(by best achievable fidelity)',
                  fontsize=10, fontweight='bold', color=DARK, pad=8)

    pie_leg = [mpatches.Patch(color=c, alpha=0.85, label=l.split('\n')[0])
               for c, l in zip(pie_colors, labels_pie)]
    ax3.legend(handles=pie_leg, fontsize=8.5, framealpha=0.95,
               edgecolor='#CCC', loc='lower center',
               bbox_to_anchor=(0.5, -0.12), ncol=3)

    fig.suptitle('Quantum Link Comparison Dashboard -- BTS Backbone vs Mobile Nodes',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, "fig13c_dashboard_comparison.png")
