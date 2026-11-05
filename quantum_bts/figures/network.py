from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from ..channel import transmittance
from ..config import MAX_LINK_KM, MAX_MOBILE_KM
from .style import (
    ax_style, BLUE, cmap_fid, DARK, GREEN, ORANGE, PINK, PURPLE, RADIO_COLORS, RED,
    save_fig, TEAL,
)


def fig_network_map(G, df_raw, clusters, mobile_df, assignments, mob_links):
    fig, ax = plt.subplots(figsize=(15,12))
    fig.patch.set_facecolor('white')
    ax_style(ax)

    for radio, grp in df_raw.groupby('radio'):
        ax.scatter(grp['lon'],grp['lat'], s=5, c=RADIO_COLORS[radio],
                   alpha=0.2, zorder=1, label=f'{radio} cells ({len(grp)})')

    edges = list(G.edges(data=True))
    fvals = np.array([e[2]['fidelity'] for e in edges]) if edges else np.array([0.5])
    for u,v,data in edges:
        pu=G.nodes[u]['pos']; pv=G.nodes[v]['pos']
        norm=(data['fidelity']-fvals.min())/(fvals.max()-fvals.min()+1e-9)
        col=cmap_fid(norm)
        lw=0.8+3.5*(1-data['distance_km']/MAX_LINK_KM)
        ax.plot([pu[0],pv[0]],[pu[1],pv[1]], color=col,lw=lw,alpha=0.7,zorder=2)

    for _,row in mob_links.iterrows():
        bts_node = G.nodes.get(row['bts_name'])
        if bts_node:
            ax.plot([row['mob_lon'], bts_node['pos'][0]],
                    [row['mob_lat'], bts_node['pos'][1]],
                    color=PINK, lw=0.5, alpha=0.3, zorder=2)

    type_styles = {
        'hub'  : dict(s=320, marker='o', ec=DARK,   lw=1.8, c=BLUE),
        'relay': dict(s=180, marker='s', ec=DARK,   lw=1.4, c=TEAL),
        'node' : dict(s=80,  marker='^', ec='#777', lw=1.0, c=ORANGE),
    }
    for node, nd in G.nodes(data=True):
        st = type_styles[nd['ntype']]
        ax.scatter(*nd['pos'], zorder=5, **st)
        txt = ax.text(nd['pos'][0], nd['pos'][1]+0.003, node,
                      fontsize=6.5, ha='center', color=DARK, zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

    mob_type_col = {'pedestrian': GREEN, 'vehicle': ORANGE, 'IoT': PURPLE}
    for _,mob in mobile_df.iterrows():
        col = mob_type_col[mob['mob_type']]
        ax.scatter(mob['lon'], mob['lat'], s=18, c=col,
                   marker='D', alpha=0.7, edgecolors='none', zorder=4)

    leg = [
        mpatches.Patch(color=RADIO_COLORS['GSM'],  label='GSM cells'),
        mpatches.Patch(color=RADIO_COLORS['UMTS'], label='UMTS cells'),
        mpatches.Patch(color=RADIO_COLORS['LTE'],  label='LTE cells'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=BLUE,  markersize=10,markeredgecolor=DARK,label='Hub BTS'),
        Line2D([0],[0],marker='s',color='w',markerfacecolor=TEAL,  markersize=9, markeredgecolor=DARK,label='Relay BTS'),
        Line2D([0],[0],marker='^',color='w',markerfacecolor=ORANGE,markersize=8, markeredgecolor='#777',label='Node BTS'),
        Line2D([0],[0],marker='D',color='w',markerfacecolor=GREEN,  markersize=7,markeredgecolor='none',label='Mobile (pedestrian)'),
        Line2D([0],[0],marker='D',color='w',markerfacecolor=ORANGE, markersize=7,markeredgecolor='none',label='Mobile (vehicle)'),
        Line2D([0],[0],marker='D',color='w',markerfacecolor=PURPLE, markersize=7,markeredgecolor='none',label='Mobile (IoT)'),
        Line2D([0],[0],color=PINK,lw=1.5,alpha=0.8,label='Mobile FSO link'),
    ]
    ax.legend(handles=leg, loc='lower right', fontsize=7.5, framealpha=0.95,
              edgecolor='#CCC', fancybox=True, ncol=2)

    sm = plt.cm.ScalarMappable(cmap=cmap_fid, norm=Normalize(vmin=fvals.min(),vmax=fvals.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=35)
    cbar.set_label('BTS Link Fidelity F(Φ⁺)', fontsize=9, color=DARK)

    ax.set_xlabel('Longitude (°E)', fontsize=11, color=DARK)
    ax.set_ylabel('Latitude (°N)',  fontsize=11, color=DARK)
    ax.set_title('Quantum BTS + Mobile Node Network -- Kathmandu Valley\n'
                 '810 nm FSO Entanglement Links | Mobile Range <= 4500 m',
                 fontsize=13, fontweight='bold', color=DARK, pad=12)

    ax.text(0.02,0.98,
        f"BTS Sites: {len(clusters)} | Mobile Nodes: {len(mobile_df)}\n"
        f"BTS Links: {len(edges)} | Mobile Links: {len(mob_links)}",
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(boxstyle='round,pad=0.5',fc='white',ec='#CBD5E1',alpha=0.95), color=DARK)

    return save_fig(fig, "fig01_network_map.png")


def fig_mobile_spatial(mobile_df, clusters):
    fig, axes = plt.subplots(1,2, figsize=(16,7))
    fig.patch.set_facecolor('white')

    ax = axes[0]; ax_style(ax, '(A)  Mobile Node Positions -- Speed')
    mob_type_markers = {'pedestrian':'o', 'vehicle':'s', 'IoT':'^'}
    for mtype, grp in mobile_df.groupby('mob_type'):
        sc = ax.scatter(grp['lon'], grp['lat'],
                        c=grp['speed_kmh'], cmap='plasma',
                        vmin=0, vmax=80,
                        s=50, alpha=0.85, marker=mob_type_markers[mtype],
                        edgecolors='white', lw=0.4, zorder=4,
                        label=mtype)
    for _,bts in clusters.iterrows():
        circle = plt.Circle((bts['lon'],bts['lat']),
                             MAX_MOBILE_KM/111.0, color=BLUE,
                             fill=False, lw=0.5, alpha=0.2, zorder=1)
        ax.add_patch(circle)
        ax.scatter(bts['lon'],bts['lat'], s=80, c=BLUE,
                   marker='*', edgecolors=DARK, lw=0.8, zorder=5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_label('Speed (km/h)', fontsize=9, color=DARK)
    ax.set_xlabel('Longitude (°E)', fontsize=10, color=DARK)
    ax.set_ylabel('Latitude (°N)',  fontsize=10, color=DARK)
    ax.legend(fontsize=8.5, framealpha=0.9)

    ax2 = axes[1]; ax_style(ax2, '(B)  Speed Distribution by Mobile Type')
    for mtype, grp in mobile_df.groupby('mob_type'):
        ax2.hist(grp['speed_kmh'], bins=15, alpha=0.65, label=mtype, edgecolor='white')
    ax2.set_xlabel('Speed (km/h)', fontsize=10, color=DARK)
    ax2.set_ylabel('Count', fontsize=10, color=DARK)
    ax2.legend(fontsize=9)

    fig.suptitle('Mobile Node Spatial Distribution & Speed Profile', fontsize=13,
                 fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig02_mobile_spatial.png")


def fig_coverage_map(mob_links, mobile_df, clusters):
    fig, axes = plt.subplots(1,2, figsize=(17,7))
    fig.patch.set_facecolor('white')

    best_fid = mob_links.groupby('mobile_id')['fidelity'].max().reset_index()
    best_fid.columns = ['mobile_id','best_fidelity']
    mob_merged = mobile_df.merge(best_fid, on='mobile_id', how='left')
    mob_merged['best_fidelity'] = mob_merged['best_fidelity'].fillna(0)

    ax1 = axes[0]
    ax_style(ax1, '(A)  Best Fidelity per Mobile Node')
    sc = ax1.scatter(mob_merged['lon'], mob_merged['lat'],
                     c=mob_merged['best_fidelity'], cmap=cmap_fid,
                     vmin=0, vmax=1, s=60, alpha=0.88,
                     edgecolors=DARK, lw=0.5, zorder=4,
                     label='Mobile node (colour = best F)')
    bts_sc = ax1.scatter(clusters['lon'], clusters['lat'],
                         s=120, c=BLUE, marker='*',
                         edgecolors=DARK, lw=0.9, zorder=5,
                         label='BTS site (★)')
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.04, pad=0.01)
    cbar.set_label('Best Fidelity F(Φ⁺)', fontsize=9, color=DARK)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.axhline(0.5,  color=RED,   lw=1.8, ls='--')
    cbar.ax.axhline(2/3,  color=GREEN, lw=1.8, ls='--')
    cbar.ax.text(1.08, 0.5,  'F=0.5\n(Ent.)',  transform=cbar.ax.transAxes,
                 fontsize=6.5, color=RED,   va='center')
    cbar.ax.text(1.08, 2/3,  'F=2/3\n(QKD)',   transform=cbar.ax.transAxes,
                 fontsize=6.5, color=GREEN, va='center')

    legend_items = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=DARK,
               markersize=8, markeredgecolor=DARK, label='Mobile node'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor=BLUE,
               markersize=12, markeredgecolor=DARK, label='BTS site (★)'),
    ]
    ax1.legend(handles=legend_items, fontsize=8.5, framealpha=0.95,
               edgecolor='#CCC', loc='lower right')
    ax1.set_xlabel('Longitude (°E)', fontsize=10, color=DARK)
    ax1.set_ylabel('Latitude (°N)',  fontsize=10, color=DARK)

    ax2 = axes[1]
    ax_style(ax2, '(B)  Entanglement Coverage Classification')

    entangled = mob_merged['best_fidelity'] > 0.5
    qkd_ready = mob_merged['best_fidelity'] > 2/3
    no_link   = mob_merged['best_fidelity'] == 0

    cat_data = [
        (mob_merged[no_link],           RED,    30, 'X', 'No BTS in range (F=0)'),
        (mob_merged[~entangled & ~no_link], '#FF7043', 35, 'v', 'Sub-threshold  (0<F<=0.5)'),
        (mob_merged[entangled & ~qkd_ready], ORANGE, 45, 'D', 'Entangled  (0.5<F<=0.667)'),
        (mob_merged[qkd_ready],         GREEN,  55, 'o', 'QKD-ready  (F>0.667)'),
    ]
    for grp_df, col, sz, mk, lbl in cat_data:
        if len(grp_df):
            ax2.scatter(grp_df['lon'], grp_df['lat'],
                        c=col, s=sz, marker=mk, alpha=0.85,
                        edgecolors=DARK, lw=0.5, zorder=4, label=lbl)

    ax2.scatter(clusters['lon'], clusters['lat'],
                s=130, c=BLUE, marker='*',
                edgecolors=DARK, lw=0.9, zorder=6,
                label='BTS site (★)')

    n_tot = len(mob_merged)
    n_ent = int(entangled.sum()); n_qkd = int(qkd_ready.sum())
    n_sub = int((~entangled & ~no_link).sum()); n_none = int(no_link.sum())
    ax2.text(0.02, 0.98,
             f"Total mobile nodes : {n_tot}\n"
             f"QKD-ready (F>2/3)  : {n_qkd}  ({n_qkd/n_tot*100:.1f}%)\n"
             f"Entangled (F>0.5)  : {n_ent}  ({n_ent/n_tot*100:.1f}%)\n"
             f"Sub-threshold       : {n_sub}  ({n_sub/n_tot*100:.1f}%)\n"
             f"No BTS in range    : {n_none}  ({n_none/n_tot*100:.1f}%)",
             transform=ax2.transAxes, fontsize=8.2, va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='white',
                       ec='#CBD5E1', alpha=0.95), color=DARK)

    ax2.legend(fontsize=8.5, framealpha=0.95, edgecolor='#CCC',
               loc='lower right', title='Coverage class', title_fontsize=8)
    ax2.set_xlabel('Longitude (°E)', fontsize=10, color=DARK)
    ax2.set_ylabel('Latitude (°N)',  fontsize=10, color=DARK)

    fig.suptitle('Mobile Node Entanglement Coverage Map  -- Kathmandu Valley',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig06_coverage_map.png")


def fig_bts_load_count(mob_links):
    load = mob_links.groupby('bts_name').agg(
        n_mobiles=('mobile_id','nunique'),
        mean_fidelity=('fidelity','mean'),
        mean_qber=('qber','mean'),
        mean_conc=('concurrence','mean'),
    ).reset_index().sort_values('n_mobiles', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(7, len(load)*0.42)))
    fig.patch.set_facecolor('white')
    ax_style(ax, 'Mobile Nodes Served per BTS Site  (colour = mean fidelity)')

    colors_bar = [cmap_fid(f) for f in load['mean_fidelity']]
    bars = ax.barh(range(len(load)), load['n_mobiles'],
                   color=colors_bar, edgecolor=DARK, lw=0.8, alpha=0.88)

    for i, (bar, row) in enumerate(zip(bars, load.itertuples())):
        w = bar.get_width()
        ax.text(w + 0.08, bar.get_y()+bar.get_height()/2,
                f'{int(row.n_mobiles)}  (F̄={row.mean_fidelity:.3f})',
                va='center', fontsize=8.5, color=DARK, fontweight='bold')

    ax.set_yticks(range(len(load)))
    ax.set_yticklabels(load['bts_name'], fontsize=8.5, color=DARK)
    ax.set_xlabel('Number of Mobile Nodes Served', fontsize=11, color=DARK)
    ax.set_xlim(right=load['n_mobiles'].max() * 1.25)

    sm = plt.cm.ScalarMappable(cmap=cmap_fid,
         norm=Normalize(vmin=load['mean_fidelity'].min(),
                        vmax=load['mean_fidelity'].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=35)
    cbar.set_label('Mean Fidelity F(Φ⁺)', fontsize=9, color=DARK)
    cbar.ax.tick_params(labelsize=8)

    leg = [
        mpatches.Patch(color=cmap_fid(0.9), alpha=0.88, label='High fidelity BTS'),
        mpatches.Patch(color=cmap_fid(0.5), alpha=0.88, label='Medium fidelity BTS'),
        mpatches.Patch(color=cmap_fid(0.2), alpha=0.88, label='Low fidelity BTS'),
    ]
    ax.legend(handles=leg, fontsize=8.5, framealpha=0.95, edgecolor='#CCC',
              loc='lower right', title='Fidelity colour guide', title_fontsize=8)

    fig.suptitle('BTS Load Analysis -- Mobile Nodes Served per Site',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig07a_bts_load_count.png")


def fig_bts_load_metrics(mob_links):
    load = mob_links.groupby('bts_name').agg(
        n_mobiles=('mobile_id','nunique'),
        mean_fidelity=('fidelity','mean'),
        std_fidelity=('fidelity','std'),
        mean_qber=('qber','mean'),
        std_qber=('qber','std'),
        mean_conc=('concurrence','mean'),
    ).reset_index().sort_values('mean_fidelity', ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    fig.patch.set_facecolor('white')

    x = np.arange(len(load))
    w = 0.38

    ax1 = axes[0]
    ax_style(ax1, 'Mean Fidelity per BTS Site  (serving mobile nodes)')
    bars_f = ax1.bar(x, load['mean_fidelity'],
                     yerr=load['std_fidelity'], capsize=4,
                     color=BLUE, alpha=0.82, edgecolor=DARK, lw=0.8,
                     width=0.6, label='Mean Fidelity F(Φ⁺)',
                     error_kw=dict(ecolor=DARK, lw=1.2, capthick=1.2))
    ax1.axhline(0.5, color=RED,   ls=':', lw=1.8,
                label='Entanglement threshold (F=0.5)')
    ax1.axhline(2/3, color=GREEN, ls=':', lw=1.8,
                label='QKD threshold (F=0.667)')
    for bar, val in zip(bars_f, load['mean_fidelity']):
        ax1.text(bar.get_x()+bar.get_width()/2, val+0.015,
                 f'{val:.3f}', ha='center', fontsize=7.5,
                 fontweight='bold', color=DARK)
    ax1.set_ylabel('Mean Fidelity F(Φ⁺)', fontsize=10, color=DARK)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC',
               loc='lower right', ncol=3)

    ax2 = axes[1]
    ax_style(ax2, 'Mean QBER per BTS Site  (serving mobile nodes)')
    bars_q = ax2.bar(x, load['mean_qber'],
                     yerr=load['std_qber'], capsize=4,
                     color=RED, alpha=0.82, edgecolor=DARK, lw=0.8,
                     width=0.6, label='Mean QBER',
                     error_kw=dict(ecolor=DARK, lw=1.2, capthick=1.2))
    ax2.axhline(0.11, color=ORANGE, ls=':', lw=1.8,
                label='Acceptable QBER threshold (0.11)')
    ax2.axhline(0.25, color=RED,    ls='--', lw=1.5, alpha=0.6,
                label='Critical QBER limit (0.25)')
    for bar, val in zip(bars_q, load['mean_qber']):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.004,
                 f'{val:.3f}', ha='center', fontsize=7.5,
                 fontweight='bold', color=DARK)
    ax2.set_ylabel('Mean QBER', fontsize=10, color=DARK)
    ax2.set_ylim(0, load['mean_qber'].max() * 1.35)
    ax2.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC',
               loc='upper right', ncol=2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(load['bts_name'], rotation=40, ha='right',
                        fontsize=8, color=DARK)
    ax2.set_xlabel('BTS Site', fontsize=11, color=DARK)

    fig.suptitle('BTS Performance Metrics -- Fidelity & QBER for Mobile Links',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig07b_bts_load_metrics.png")


def fig_fidelity_heatmap(G, clusters):
    names = [r['name'] for _,r in clusters.iterrows() if r['name'] in G.nodes]
    n = len(names)
    mat = np.full((n,n), np.nan)
    for i,u in enumerate(names):
        for j,v in enumerate(names):
            if G.has_edge(u,v):
                mat[i,j] = G[u][v]['fidelity']
            elif i==j:
                mat[i,j] = 1.0

    fig, ax = plt.subplots(figsize=(13,11))
    fig.patch.set_facecolor('white')
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap_fid, vmin=0, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, label='Fidelity F(Φ⁺)')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6.5, color=DARK)
    ax.set_yticklabels(names, fontsize=6.5, color=DARK)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i,j]):
                ax.text(j,i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=5.5, color='white' if mat[i,j]<0.6 else DARK)
    ax.set_title('BTS-to-BTS Link Fidelity Matrix', fontsize=13,
                 fontweight='bold', color=DARK, pad=12)
    plt.tight_layout()
    return save_fig(fig, "fig10_fidelity_heatmap.png")


def fig_transmittance(mob_links, G):
    bts_edges = [(d['distance_km'], d['transmittance']) for _,_,d in G.edges(data=True)]

    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('white'); ax_style(ax, 'Transmittance vs Distance')

    if bts_edges:
        bd, bt = zip(*bts_edges)
        ax.scatter(bd, bt, c=BLUE, s=80, alpha=0.8, label='BTS↔BTS links',
                   edgecolors=DARK, lw=0.7, marker='o', zorder=4)
    ax.scatter(mob_links['distance_km'], mob_links['transmittance'],
               c=PINK, s=30, alpha=0.6, label='Mobile↔BTS links',
               edgecolors='none', marker='D', zorder=3)

    ds = np.linspace(0.1, MAX_LINK_KM, 200)
    ax.plot(ds, [transmittance(d) for d in ds], color=DARK, lw=2, ls='--',
            label='Theoretical T(d)', zorder=5)
    ax.set_xlabel('Link Distance (km)', fontsize=11, color=DARK)
    ax.set_ylabel('Transmittance T', fontsize=11, color=DARK)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save_fig(fig, "fig11_transmittance.png")
