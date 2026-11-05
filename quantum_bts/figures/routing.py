from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from ..config import QKD_FIDELITY_THRESH, SWAP_FIDELITY_THRESH
from .style import (
    ax_style, BLUE, cmap_fid, DARK, GRAY, GREEN, ORANGE, PURPLE, RED, save_fig, TEAL,
)
from ..swapping import swap_fidelity


def fig_routing_heatmap(routing_table, G):
    nodes = sorted(set(routing_table['source'].tolist() +
                        routing_table['destination'].tolist()))
    n = len(nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}
    mat = np.full((n, n), np.nan)
    for _, row in routing_table.iterrows():
        i, j = idx[row['source']], idx[row['destination']]
        mat[i, j] = row['e2e_fidelity']
        mat[j, i] = row['e2e_fidelity']
    np.fill_diagonal(mat, 1.0)

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor('white')
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap_fid, vmin=0, vmax=1, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, aspect=35)
    cbar.set_label('E2E Fidelity after Entanglement Swapping', fontsize=9, color=DARK)
    cbar.ax.axhline(0.5,  color=RED,   lw=1.5, ls='--')
    cbar.ax.axhline(2/3,  color=GREEN, lw=1.5, ls='--')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short_names = [nd[:15] for nd in nodes]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6.5, color=DARK)
    ax.set_yticklabels(short_names, fontsize=6.5, color=DARK)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                val = mat[i, j]
                fc  = 'white' if val < 0.55 else DARK
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=5.5, color=fc, fontweight='bold')

    ax.set_title('Dijkstra Entanglement-Routing -- E2E Fidelity Matrix\n'
                 '(Optimal swap-path fidelity for every hub/relay pair)',
                 fontsize=12, fontweight='bold', color=DARK, pad=12)

    n_qkd = (routing_table['qkd_ready'].sum())
    n_ent = (routing_table['entangled'].sum())
    n_tot = len(routing_table)
    ax.text(0.02, 0.98,
            f"Pairs computed : {n_tot}\n"
            f"QKD-ready (F>2/3) : {n_qkd} ({n_qkd/n_tot*100:.0f}%)\n"
            f"Entangled (F>0.5)  : {n_ent} ({n_ent/n_tot*100:.0f}%)",
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white',
                      ec='#CBD5E1', alpha=0.96), color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig14a_routing_heatmap.png")


def fig_routing_gain(routing_table):
    df_gain = (routing_table
               .assign(abs_gain=routing_table['routing_gain'].abs())
               .sort_values('routing_gain', ascending=False)
               .head(20)
               .copy())
    df_gain['src_short'] = df_gain['source'].str[:14]
    df_gain['dst_short'] = df_gain['destination'].str[:14]
    df_gain['pair_label'] = df_gain['src_short'] + '\n↔  ' + df_gain['dst_short']
    df_gain = df_gain.iloc[::-1].reset_index(drop=True)
    n_gain = len(df_gain)

    fig1_h = max(8, n_gain * 0.55)
    fig1, ax1 = plt.subplots(figsize=(16, fig1_h))
    fig1.patch.set_facecolor('white')
    ax_style(ax1, None)

    y = np.arange(n_gain)
    gain_colors = [GREEN if g >= 0 else RED for g in df_gain['routing_gain']]
    hbars = ax1.barh(y, df_gain['routing_gain'],
                     color=gain_colors, edgecolor=DARK,
                     lw=0.8, alpha=0.85, height=0.60)

    for bar, val, row in zip(hbars, df_gain['routing_gain'], df_gain.itertuples()):
        sign = '+' if val >= 0 else ''
        x_txt = val + 0.0003 if val >= 0 else val - 0.0003
        ha_txt = 'left' if val >= 0 else 'right'
        ax1.text(x_txt, bar.get_y() + bar.get_height()/2,
                 f' {sign}{val:.5f}  ({int(row.hops)}h)',
                 va='center', ha=ha_txt, fontsize=8.0,
                 fontweight='bold', color=DARK)

    ax1.axvline(0, color=DARK, lw=1.4, ls='--', alpha=0.65, zorder=5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df_gain['pair_label'], fontsize=9.0, color=DARK)
    ax1.set_xlabel('Routing Gain  ΔF = F(Dijkstra) − F(Fewest-hop)',
                   fontsize=11, color=DARK)
    ax1.set_ylim(-0.7, n_gain - 0.3)

    leg1 = [
        mpatches.Patch(color=GREEN, alpha=0.85,
                       label='Dijkstra improves on fewest-hop  (+ΔF)'),
        mpatches.Patch(color=RED,   alpha=0.85,
                       label='Fewest-hop already optimal  (ΔF = 0 or −)'),
    ]
    ax1.legend(handles=leg1, fontsize=9.5, framealpha=0.95,
               edgecolor='#CCC', loc='lower right')

    pos_gains = df_gain[df_gain['routing_gain'] > 0]['routing_gain']
    ax1.text(0.02, 0.98,
             f"Pairs shown  : {n_gain}\n"
             f"With +ΔF     : {len(pos_gains)}\n"
             f"Max ΔF       : {df_gain['routing_gain'].max():+.5f}\n"
             f"Mean ΔF      : {df_gain['routing_gain'].mean():+.5f}",
             transform=ax1.transAxes, fontsize=8.5, va='top',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', fc='white',
                       ec='#CBD5E1', alpha=0.96), color=DARK)

    fig1.suptitle('Dijkstra Routing Gain vs Fewest-Hop Baseline\n'
                  '(ΔF > 0 means Dijkstra finds a higher-fidelity path)',
                  fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    path1 = save_fig(fig1, "fig14b1_routing_gain_bars.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
    fig2.patch.set_facecolor('white')

    ax2a = axes2[0]
    ax_style(ax2a, '(A)  E2E Fidelity vs Number of Hops  (all routed pairs)')
    sc = ax2a.scatter(
        routing_table['hops'], routing_table['e2e_fidelity'],
        c=routing_table['total_dist_km'], cmap='plasma_r',
        s=55 + routing_table['n_swaps'] * 25,
        alpha=0.82, edgecolors=DARK, lw=0.6, zorder=4)
    cbar2 = fig2.colorbar(sc, ax=ax2a, fraction=0.04, pad=0.01)
    cbar2.set_label('Total path distance (km)', fontsize=9, color=DARK)
    ax2a.axhline(0.5, color=RED,   ls=':', lw=1.8,
                 label='Entanglement threshold (F=0.5)')
    ax2a.axhline(2/3, color=GREEN, ls=':', lw=1.8,
                 label='QKD threshold (F=0.667)')
    ax2a.set_xlabel('Number of hops  (swaps = hops − 1)', fontsize=10, color=DARK)
    ax2a.set_ylabel('E2E Fidelity F(Φ⁺)', fontsize=10, color=DARK)
    ax2a.set_xlim(left=0.5)
    ax2a.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC', loc='upper right')
    for sw, lbl in [(0,'0 swaps'), (1,'1 swap'), (3,'3 swaps')]:
        ax2a.scatter([], [], s=55+sw*25, c=GRAY, alpha=0.7,
                     edgecolors=DARK, lw=0.6, label=f'Size: {lbl}')
    ax2a.legend(fontsize=8.5, framealpha=0.95, edgecolor='#CCC',
                loc='upper right', ncol=1,
                title='Thresholds / swap count', title_fontsize=8)

    ax2b = axes2[1]
    ax_style(ax2b, '(B)  Dijkstra E2E F vs Fewest-Hop E2E F')
    col_pt = [GREEN if g > 0.001 else (ORANGE if g > 0 else BLUE)
              for g in routing_table['routing_gain']]
    ax2b.scatter(routing_table['fewest_hop_e2e'],
                 routing_table['e2e_fidelity'],
                 c=col_pt, s=50, alpha=0.80,
                 edgecolors=DARK, lw=0.5, zorder=4)
    lim = [0, 1.02]
    ax2b.plot(lim, lim, color=DARK, lw=1.5, ls='--', alpha=0.55,
              label='Identity  (Dijkstra = fewest-hop)')
    ax2b.axhline(0.5, color=RED,   ls=':', lw=1.5)
    ax2b.axhline(2/3, color=GREEN, ls=':', lw=1.5)
    ax2b.axvline(0.5, color=RED,   ls=':', lw=1.5)
    ax2b.axvline(2/3, color=GREEN, ls=':', lw=1.5)
    ax2b.set_xlim(0, 1.05); ax2b.set_ylim(0, 1.05)
    ax2b.set_xlabel('Fewest-Hop E2E Fidelity F(Φ⁺)', fontsize=10, color=DARK)
    ax2b.set_ylabel('Dijkstra E2E Fidelity F(Φ⁺)',   fontsize=10, color=DARK)
    leg2b = [
        mpatches.Patch(color=GREEN,  alpha=0.85, label='Dijkstra gain > 0.001'),
        mpatches.Patch(color=ORANGE, alpha=0.85, label='Marginal gain (0-0.001)'),
        mpatches.Patch(color=BLUE,   alpha=0.85, label='No gain (same path)'),
        Line2D([0],[0], color=DARK, ls='--', lw=1.5, alpha=0.55,
               label='Identity line'),
    ]
    ax2b.legend(handles=leg2b, fontsize=8.5, framealpha=0.95,
                edgecolor='#CCC', loc='upper left')

    fig2.suptitle('Entanglement-Routing Performance -- Dijkstra vs Greedy Baseline',
                  fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    path2 = save_fig(fig2, "fig14b2_routing_scatter.png")
    return path1, path2


def fig_routing_all_paths(G, all_paths, clusters):
    pos = {n: G.nodes[n]['pos'] for n in G.nodes()}

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.patch.set_facecolor('white')

    for ax_i, (ax, filt_fn, title_sfx) in enumerate([
        (axes[0], lambda p: p['e2e_fidelity'] >= QKD_FIDELITY_THRESH,
         'QKD-ready paths  (E2E F >= 0.667)'),
        (axes[1], lambda p: p['e2e_fidelity'] < QKD_FIDELITY_THRESH,
         'Sub-QKD paths  (E2E F < 0.667)'),
    ]):
        ax_style(ax, f'({chr(65+ax_i)})  {title_sfx}')

        for u, v, d in G.edges(data=True):
            pu, pv = pos[u], pos[v]
            ax.plot([pu[0],pv[0]], [pu[1],pv[1]],
                    color='#E2E8F0', lw=0.8, alpha=0.5, zorder=1)

        filtered = [p for p in all_paths if filt_fn(p)]
        for path_info in filtered:
            pnodes = path_info['path_nodes']
            e2e_F  = path_info['e2e_fidelity']
            col    = cmap_fid(e2e_F)
            alpha  = 0.35 + 0.55 * e2e_F
            lw     = 1.0 + 3.0 * e2e_F
            for u, v in zip(pnodes[:-1], pnodes[1:]):
                pu, pv = pos[u], pos[v]
                ax.plot([pu[0],pv[0]], [pu[1],pv[1]],
                        color=col, lw=lw, alpha=alpha, zorder=3,
                        solid_capstyle='round')

        type_styles = {'hub'  :(BLUE,250,'o'),
                       'relay':(TEAL,150,'s'),
                       'node' :(ORANGE,60,'^')}
        for node, nd in G.nodes(data=True):
            col_n, sz, mk = type_styles[nd['ntype']]
            ax.scatter(*pos[node], s=sz, c=col_n, marker=mk,
                       edgecolors=DARK, lw=0.9, zorder=5)
            if nd['ntype'] in ('hub', 'relay'):
                txt = ax.text(pos[node][0], pos[node][1]+0.003, node,
                              fontsize=6, ha='center', color=DARK, zorder=7)
                txt.set_path_effects([pe.withStroke(linewidth=2.5,
                                                     foreground='white')])

        sm = plt.cm.ScalarMappable(cmap=cmap_fid, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=35)
        cbar.set_label('E2E Fidelity', fontsize=9, color=DARK)
        cbar.ax.axhline(2/3, color=GREEN, lw=1.5, ls='--')
        cbar.ax.axhline(0.5, color=RED,   lw=1.5, ls='--')

        leg_items = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor=BLUE,
                   markersize=10, markeredgecolor=DARK, label='Hub BTS'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor=TEAL,
                   markersize=9,  markeredgecolor=DARK, label='Relay BTS'),
            Line2D([0],[0], marker='^', color='w', markerfacecolor=ORANGE,
                   markersize=8,  markeredgecolor=DARK, label='Node BTS'),
            Line2D([0],[0], color=cmap_fid(0.9), lw=3, alpha=0.9,
                   label='High-fidelity path'),
            Line2D([0],[0], color=cmap_fid(0.4), lw=1.5, alpha=0.6,
                   label='Lower-fidelity path'),
        ]
        ax.legend(handles=leg_items, fontsize=8, framealpha=0.95,
                  edgecolor='#CCC', loc='lower right',
                  title=f'Paths shown: {len(filtered)}', title_fontsize=8)
        ax.set_xlabel('Longitude (°E)', fontsize=10, color=DARK)
        ax.set_ylabel('Latitude (°N)',  fontsize=10, color=DARK)

    fig.suptitle('All Dijkstra-Optimal Entanglement Paths -- Kathmandu BTS Network\n'
                 '(line colour & thickness ∝ E2E fidelity)',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig14c_all_routing_paths.png")


def fig_routing_hop_profiles(all_paths, top_n=8):
    top = sorted(all_paths, key=lambda x: -x['e2e_fidelity'])[:top_n]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=False)
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    for ax_i, (path_info, ax) in enumerate(zip(top, axes_flat)):
        ax_style(ax, None)
        hops     = path_info['hop_fids']
        pnodes   = path_info['path_nodes']
        e2e_F    = path_info['e2e_fidelity']
        n_hops   = len(hops)
        x        = np.arange(1, n_hops + 1)

        acc_F = [hops[0]]
        for f in hops[1:]:
            acc_F.append(swap_fidelity(acc_F[-1], f))

        colors_hop = [cmap_fid(f) for f in hops]
        bars = ax.bar(x, hops, color=colors_hop, edgecolor=DARK,
                      lw=0.8, alpha=0.85, width=0.55, label='Per-hop F')
        ax.plot(x, acc_F, color=DARK, lw=2.2, ls='--', marker='D',
                markersize=7, markerfacecolor=BLUE,
                markeredgecolor=DARK, label='Accumulated E2E F')
        ax.axhline(0.5, color=RED,   ls=':', lw=1.5)
        ax.axhline(2/3, color=GREEN, ls=':', lw=1.5)

        hop_labels = [f"Hop {i+1}\n{pnodes[i][:8]}->{pnodes[i+1][:8]}"
                      for i in range(n_hops)]
        ax.set_xticks(x)
        ax.set_xticklabels(hop_labels, fontsize=5.5, color=DARK)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Fidelity F(Φ⁺)', fontsize=8, color=DARK)

        ax.set_title(
            f'Path {ax_i+1}  |  E2E F = {e2e_F:.4f}\n'
            f'{pnodes[0][:12]} -> … -> {pnodes[-1][:12]}',
            fontsize=8, fontweight='bold', color=DARK, pad=5)

        if ax_i == 0:
            ax.legend(fontsize=7.5, framealpha=0.92, edgecolor='#CCC',
                      loc='upper right')

        ax.text(n_hops, acc_F[-1] + 0.04,
                f'E2E={e2e_F:.3f}', ha='center', fontsize=7.5,
                fontweight='bold', color=BLUE,
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec=BLUE, alpha=0.9))

    fig.suptitle(f'Per-Hop Fidelity Profile -- Top {top_n} Dijkstra-Optimal Paths\n'
                 '(bars = individual hop F | dashed = accumulated swap fidelity)',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, "fig14d_hop_profiles.png")


def fig_purification_comparison(routing_table):
    rt = routing_table.dropna(subset=['purified_e2e_fidelity']).copy()
    if rt.empty:
        print("  [!] No purification data available, skipping fig14e.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.patch.set_facecolor('white')

    ax = axes[0, 0]
    ax_style(ax, '(A)  Raw vs. Purified End-to-End Fidelity')
    sc = ax.scatter(rt['e2e_fidelity'], rt['purified_e2e_fidelity'],
                    c=rt['hops'], cmap='viridis', s=60,
                    edgecolor=DARK, lw=0.6, alpha=0.85, zorder=3)
    lims = [min(rt['e2e_fidelity'].min(), rt['purified_e2e_fidelity'].min()) - 0.02,
            max(rt['e2e_fidelity'].max(), rt['purified_e2e_fidelity'].max()) + 0.02]
    ax.plot(lims, lims, color=DARK, ls='--', lw=1.3, alpha=0.6, zorder=2,
            label='No change (y = x)')
    ax.axvline(SWAP_FIDELITY_THRESH, color=RED, ls=':', lw=1.3, alpha=0.7)
    ax.axhline(SWAP_FIDELITY_THRESH, color=RED, ls=':', lw=1.3, alpha=0.7)
    ax.axvline(QKD_FIDELITY_THRESH, color=GREEN, ls=':', lw=1.3, alpha=0.7)
    ax.axhline(QKD_FIDELITY_THRESH, color=GREEN, ls=':', lw=1.3, alpha=0.7)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Hops', fontsize=9, color=DARK)
    ax.set_xlabel('Raw E2E Fidelity (post-swap)', fontsize=10, color=DARK)
    ax.set_ylabel('Purified E2E Fidelity', fontsize=10, color=DARK)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=8.5, framealpha=0.95, edgecolor='#CCC', loc='upper left')

    ax = axes[0, 1]
    ax_style(ax, '(B)  Purification Gain Distribution')
    gains = rt['purification_gain']
    ax.hist(gains, bins=20, color=TEAL, edgecolor=DARK, lw=0.6, alpha=0.82, zorder=3)
    ax.axvline(0, color=DARK, ls='--', lw=1.4, alpha=0.7)
    ax.axvline(gains.mean(), color=ORANGE, ls='-', lw=2,
              label=f'Mean = {gains.mean():+.4f}')
    ax.set_xlabel('Purification Gain  ΔF = F(purified) − F(raw)', fontsize=10, color=DARK)
    ax.set_ylabel('Number of Routed Pairs', fontsize=10, color=DARK)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC')

    ax = axes[1, 0]
    ax_style(ax, '(C)  Purification Attempt Rate by Path Length')
    rt['hop_bucket'] = rt['hops'].clip(upper=5)
    grp = rt.groupby('hop_bucket')['purification_attempted_frac'].mean()
    bucket_labels = [f"{int(h)}" if h < 5 else "5+" for h in grp.index]
    bars = ax.bar(bucket_labels, grp.values, color=PURPLE, edgecolor=DARK,
                  lw=0.8, alpha=0.85, zorder=3)
    for b, v in zip(bars, grp.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.1%}',
                ha='center', fontsize=9, fontweight='bold', color=DARK)
    ax.set_xlabel('Number of Hops', fontsize=10, color=DARK)
    ax.set_ylabel('Mean Fraction of Realizations Purified', fontsize=10, color=DARK)
    ax.set_ylim(0, 1.05)

    ax = axes[1, 1]
    ax_style(ax, '(D)  Purification Resource Cost: Yield vs. Gain')
    attempted = rt[rt['purification_attempted_frac'] > 0]
    if not attempted.empty:
        sc2 = ax.scatter(attempted['purification_gain'],
                         attempted['purification_yield_mean'],
                         c=attempted['purification_rounds_mean'], cmap='plasma',
                         s=70, edgecolor=DARK, lw=0.6, alpha=0.85, zorder=3)
        cbar2 = fig.colorbar(sc2, ax=ax, fraction=0.04, pad=0.02)
        cbar2.set_label('Mean Rounds Used', fontsize=9, color=DARK)
    ax.set_xlabel('Purification Gain  ΔF', fontsize=10, color=DARK)
    ax.set_ylabel('Cumulative Yield (resource survival probability)', fontsize=10, color=DARK)
    ax.set_ylim(0, 1.05)

    fig.suptitle('Entanglement Purification: Raw vs. Purified End-to-End Fidelity\n'
                 '(Adaptive DEJMPS / BBPSSW-twirled protocol selection -- see purify_adaptive())',
                 fontsize=13.5, fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig14e_purification_comparison.png")
