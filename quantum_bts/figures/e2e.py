from itertools import combinations

from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .style import (
    ax_style, BLUE, cmap_fid, DARK, GREEN, ORANGE, PURPLE, RED, save_fig, TEAL,
)
from ..swapping import real_e2e_fidelity_for_path


def fig_e2e_bar(G, pair_data):
    show = pair_data[:20]
    show_rev = list(reversed(show))
    n       = len(show_rev)
    e2e_vals = [p['e2e'] for p in show_rev]

    fig_h = max(9, n * 0.58)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor('white')
    ax_style(ax, None)

    y = np.arange(n)

    bars = ax.barh(y, e2e_vals,
                   color=[cmap_fid(v) for v in e2e_vals],
                   edgecolor=DARK, lw=0.8, alpha=0.88, height=0.62)

    direct_plotted = False
    for i, p in enumerate(show_rev):
        if p['direct'] is not None:
            ax.scatter(p['direct'], i, marker='D', s=70,
                       color=PURPLE, zorder=5, edgecolors=DARK, lw=0.9,
                       label='Direct single-hop F  (◆)' if not direct_plotted else '')
            direct_plotted = True

    for bar, val, p in zip(bars, e2e_vals, show_rev):
        hop_txt = f"  {p['hops']}hop{'s' if p['hops']>1 else ''} | F={val:.4f}"
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                hop_txt, va='center', ha='left', fontsize=8.0,
                fontweight='bold', color=DARK)

    ax.axvline(0.5, color=RED,   ls=':', lw=1.8,
               label='Entanglement threshold  (F = 0.50)', zorder=6)
    ax.axvline(2/3, color=GREEN, ls=':', lw=1.8,
               label='QKD threshold  (F = 0.667)', zorder=6)

    pair_labels = [f"{p['nodes'][0]}  ↔  {p['nodes'][-1]}" for p in show_rev]
    ax.set_yticks(y)
    ax.set_yticklabels(pair_labels, fontsize=8.5, color=DARK)
    ax.set_xlabel('End-to-End Fidelity F(Φ⁺)', fontsize=11, color=DARK)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.7, n - 0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap_fid, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01, aspect=40)
    cbar.set_label('E2E Fidelity', fontsize=9, color=DARK)
    cbar.ax.axhline(0.5,  color=RED,   lw=1.5, ls='--')
    cbar.ax.axhline(2/3,  color=GREEN, lw=1.5, ls='--')

    leg_items = [
        mpatches.Patch(color=cmap_fid(0.92), alpha=0.88,
                       label='QKD-ready  (F > 0.667)'),
        mpatches.Patch(color=cmap_fid(0.58), alpha=0.88,
                       label='Entangled  (0.5 < F <= 0.667)'),
        mpatches.Patch(color=cmap_fid(0.20), alpha=0.88,
                       label='Below threshold  (F <= 0.5)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor=PURPLE,
               markersize=9, markeredgecolor=DARK,
               label='Direct single-hop F  (◆)'),
        Line2D([0],[0], color=RED,   ls=':', lw=2,
               label='Entanglement threshold (F=0.50)'),
        Line2D([0],[0], color=GREEN, ls=':', lw=2,
               label='QKD threshold (F=0.667)'),
    ]
    ax.legend(handles=leg_items, fontsize=8.5, framealpha=0.95,
              edgecolor='#CCC', loc='lower right', ncol=2,
              title='Link quality / thresholds', title_fontsize=8)

    ax.set_title('End-to-End Fidelity via Entanglement Swapping -- BTS Backbone\n'
                 f'(top {n} pairs by E2E F  |  bar colour = fidelity class)',
                 fontsize=12, fontweight='bold', color=DARK, pad=10)
    plt.tight_layout()
    return save_fig(fig, "fig12a_e2e_fidelity_bar.png")


def fig_e2e_routing(G, pair_data):
    if not pair_data:
        print("  [!] No pair_data for routing map."); return None

    best = pair_data[0]
    pos  = {n: G.nodes[n]['pos'] for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor('white')
    ax_style(ax, None)

    for u, v, d in G.edges(data=True):
        pu, pv = pos[u], pos[v]
        f = d.get('fidelity', 0.5)
        ax.plot([pu[0],pv[0]], [pu[1],pv[1]],
                color='#D1D5DB', lw=0.6 + 1.5*f, alpha=0.35, zorder=1)

    path_nodes = best['nodes']
    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
    for hop_i, (u, v) in enumerate(path_edges):
        pu, pv = pos[u], pos[v]
        f_hop  = G[u][v]['fidelity']
        col    = cmap_fid(f_hop)
        ax.plot([pu[0],pv[0]], [pu[1],pv[1]],
                color=col, lw=5, alpha=0.92, zorder=3,
                solid_capstyle='round')
        ax.plot([pu[0],pv[0]], [pu[1],pv[1]],
                color=DARK, lw=6.5, alpha=0.18, zorder=2,
                solid_capstyle='round')
        mx, my = (pu[0]+pv[0])/2, (pu[1]+pv[1])/2
        dist_h = G[u][v]['distance_km']
        txt = ax.text(mx, my, f"Hop {hop_i+1}\nF={f_hop:.3f}\n{dist_h:.1f} km",
                      ha='center', va='center', fontsize=7.2,
                      fontweight='bold', color=DARK, zorder=7,
                      bbox=dict(boxstyle='round,pad=0.25',
                                fc='white', ec=col, alpha=0.9, lw=1.2))

    src_pos = pos[path_nodes[0]]
    dst_pos = pos[path_nodes[-1]]
    ax.annotate('', xy=dst_pos, xytext=src_pos,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.2,
                                connectionstyle='arc3,rad=0.25'),
                zorder=8)

    type_styles = {
        'hub'  : (BLUE,   280, 'o'),
        'relay': (TEAL,   170, 's'),
        'node' : (ORANGE,  80, '^'),
    }
    for node, nd in G.nodes(data=True):
        col, sz, mk = type_styles[nd['ntype']]
        in_path = node in path_nodes
        is_src  = node == path_nodes[0]
        is_dst  = node == path_nodes[-1]

        ax.scatter(*pos[node],
                   s=sz*(2.5 if in_path else 1),
                   c=col if in_path else '#D1D5DB',
                   marker=mk,
                   edgecolors=DARK if in_path else '#AAAAAA',
                   lw=2.0 if in_path else 0.7,
                   zorder=5 if in_path else 4)

        if in_path or nd['ntype'] == 'hub':
            fs   = 8.5 if in_path else 7
            fw   = 'bold' if in_path else 'normal'
            dy   = 0.004 if in_path else 0.0025
            prefix = ('▶ SOURCE: ' if is_src else
                      '▶ DEST:   ' if is_dst else '')
            label_text = f"{prefix}{node}"
            txt = ax.text(pos[node][0], pos[node][1]+dy,
                          label_text, fontsize=fs, ha='center',
                          fontweight=fw, color=DARK, zorder=9)
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

    sm = plt.cm.ScalarMappable(cmap=cmap_fid, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=35)
    cbar.set_label('Hop Fidelity F(Φ⁺)', fontsize=9, color=DARK)
    cbar.ax.axhline(0.5,  color=RED,   lw=1.5, ls='--')
    cbar.ax.axhline(2/3,  color=GREEN, lw=1.5, ls='--')

    leg_items = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=BLUE,
               markersize=11, markeredgecolor=DARK, label='Hub BTS (in path)'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=TEAL,
               markersize=10, markeredgecolor=DARK, label='Relay BTS (in path)'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor=ORANGE,
               markersize=9,  markeredgecolor=DARK, label='Node BTS (in path)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#D1D5DB',
               markersize=9,  markeredgecolor='#AAAAAA', label='Off-path BTS'),
        Line2D([0],[0], color=BLUE, lw=4, alpha=0.9,
               label='Optimal path (colour = hop F)'),
        Line2D([0],[0], color='#D1D5DB', lw=1.5, alpha=0.6,
               label='Other BTS links'),
    ]
    ax.legend(handles=leg_items, fontsize=8.5, framealpha=0.95,
              edgecolor='#CCC', loc='lower right',
              title='Node type / link legend', title_fontsize=8)

    hop_fs = [G[u][v]['fidelity'] for u,v in path_edges]
    total_d = sum(G[u][v]['distance_km'] for u,v in path_edges)
    ax.text(0.02, 0.98,
            f"Optimal path (best E2E fidelity):\n"
            f"  {' -> '.join(path_nodes)}\n\n"
            f"  Hops         : {len(path_edges)}\n"
            f"  E2E Fidelity : {best['e2e']:.4f}\n"
            f"  Total dist.  : {total_d:.2f} km\n"
            f"  Per-hop F    : {', '.join(f'{f:.3f}' for f in hop_fs)}",
            transform=ax.transAxes, fontsize=8.8, va='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', fc='white',
                      ec='#CBD5E1', alpha=0.96), color=DARK, zorder=10)

    ax.set_xlabel('Longitude (°E)', fontsize=11, color=DARK)
    ax.set_ylabel('Latitude (°N)',  fontsize=11, color=DARK)
    ax.set_title('Optimal Entanglement Routing -- Kathmandu BTS Network\n'
                 '(Highlighted path = highest end-to-end fidelity route)',
                 fontsize=12, fontweight='bold', color=DARK, pad=10)
    plt.tight_layout()
    return save_fig(fig, "fig12b_e2e_routing_map.png")


def fig_e2e(G, clusters):
    hubs = [n for n,d in G.nodes(data=True) if d['ntype'] in ('hub','relay')]
    if len(hubs) < 2: hubs = list(G.nodes())[:6]
    pair_data = []
    for src, dst in combinations(hubs[:8], 2):
        if not nx.has_path(G, src, dst): continue
        try:
            path = nx.shortest_path(G, src, dst,
                                    weight=lambda u,v,d: 1-d.get('fidelity',0))
        except: continue
        hops = list(zip(path[:-1],path[1:]))
        if not hops: continue
        fid_hops = [G[u][v]['fidelity'] for u,v in hops]
        direct = G[path[0]][path[-1]]['fidelity'] if G.has_edge(path[0],path[-1]) else None
        F_e2e, F_e2e_std, _ = real_e2e_fidelity_for_path(G, path, n_repeats=20)
        pair_data.append({'label': f"{src[:12]}\n↕\n{dst[:12]}",
                          'e2e': F_e2e, 'e2e_std': F_e2e_std,
                          'direct': direct,
                          'hops': len(hops), 'nodes': path})
    pair_data.sort(key=lambda x: -x['e2e'])
    if not pair_data:
        print("  [!] No E2E paths found, skipping E2E figures."); return None, None
    p1 = fig_e2e_bar(G, pair_data)
    p2 = fig_e2e_routing(G, pair_data)
    return p1, p2
