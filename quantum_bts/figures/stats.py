from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..config import RNG
from .style import (
    ax_style, BLUE, cmap_fid, DARK, GRAY, GREEN, MOB_TYPE_COLORS, ORANGE, PURPLE, RED,
    save_fig, TEAL,
)


def _fig_stats_overview(bts_df, mob_links, bts_node_df, cover_df):
    metrics = [
        ('fidelity',    'Fidelity F(Φ⁺)', BLUE,   (0, 1)),
        ('concurrence', 'Concurrence C',   TEAL,   (0, 1)),
        ('qber',        'QBER',            RED,    (0, 0.5)),
        ('transmittance','Transmittance T', GREEN,  (0, 1)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    for ax_i, (col, label, color, ylim) in enumerate(metrics):
        ax = axes_flat[ax_i]
        ax_style(ax, f'({chr(65+ax_i)})  {label} -- BTS vs Mobile (violin + strip)')

        bts_vals = bts_df[col].dropna().values if len(bts_df) else np.array([])
        mob_vals = mob_links[col].dropna().values

        data_plot = []
        labels_plot = []
        colors_plot = []
        if len(bts_vals):
            data_plot.append(bts_vals); labels_plot.append('BTS\nbackbone')
            colors_plot.append(BLUE)
        for mtype in ['pedestrian', 'vehicle', 'IoT']:
            sub = mob_links[mob_links['mob_type']==mtype][col].dropna().values
            if len(sub):
                data_plot.append(sub)
                labels_plot.append(f'Mobile\n{mtype}')
                colors_plot.append(MOB_TYPE_COLORS[mtype])

        parts = ax.violinplot(data_plot, positions=range(len(data_plot)),
                              showmedians=True, showextrema=True, widths=0.7)
        for i, (pc, col_v) in enumerate(zip(parts['bodies'], colors_plot)):
            pc.set_facecolor(col_v); pc.set_alpha(0.45)
        parts['cmedians'].set_color(DARK); parts['cmedians'].set_linewidth(2.5)
        parts['cmaxes'].set_color(DARK);   parts['cmaxes'].set_linewidth(1.2)
        parts['cmins'].set_color(DARK);    parts['cmins'].set_linewidth(1.2)
        parts['cbars'].set_color(DARK);    parts['cbars'].set_linewidth(1.0)

        for i, (vals, col_v) in enumerate(zip(data_plot, colors_plot)):
            jitter = RNG.uniform(-0.12, 0.12, len(vals))
            ax.scatter(i + jitter, vals, s=12, c=col_v, alpha=0.35,
                       edgecolors='none', zorder=4)

        for i, vals in enumerate(data_plot):
            ax.scatter(i, np.mean(vals), s=80, c='white', marker='D',
                       edgecolors=DARK, lw=1.5, zorder=6)

        ax.set_xticks(range(len(labels_plot)))
        ax.set_xticklabels(labels_plot, fontsize=9, color=DARK)
        ax.set_ylabel(label, fontsize=10, color=DARK)
        ax.set_ylim(*ylim)
        if col == 'fidelity':
            ax.axhline(0.5, color=RED,   ls=':', lw=1.5, label='Ent. threshold')
            ax.axhline(2/3, color=GREEN, ls=':', lw=1.5, label='QKD threshold')
            ax.legend(fontsize=8, framealpha=0.92, edgecolor='#CCC')

        for i, vals in enumerate(data_plot):
            ax.text(i, ylim[1]*0.98,
                    f'μ={np.mean(vals):.3f}\nσ={np.std(vals):.3f}',
                    ha='center', va='top', fontsize=7.5,
                    color=DARK, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='#CCC', alpha=0.85))

    fig.legend(
        [Line2D([0],[0], marker='D', color='w', markerfacecolor='white',
              markeredgecolor=DARK, markersize=8, label='◆ = Mean value')],
        ['◆ = Mean value'], loc='lower center', fontsize=9,
        framealpha=0.95, edgecolor='#CCC', ncol=1)

    fig.suptitle('Extended Statistics -- Violin + Strip Plot  |  BTS Backbone vs Mobile Types',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, "fig15a_stats_violin.png")


def _fig_stats_correlation(bts_df, mob_links):
    bts_core = ['fidelity', 'concurrence', 'qber',
                 'transmittance', 'distance_km',
                 'prop_latency_ms', 'T_rtt_ms', 'decoherence_time_ms',
                 'mean_attempts', 'secret_key_rate_bps',
                 'link_budget_db', 'entanglement_gen_rate']

    mob_core = ['fidelity', 'concurrence', 'qber',
                'transmittance', 'distance_km',
                'prop_latency_ms', 'T_rtt_ms', 'decoherence_time_ms',
                'mean_attempts', 'secret_key_rate_bps', 'link_budget_db',
                'doppler_shift_hz', 'handover_prob',
                'effective_throughput_bps']

    label_map = {
        'fidelity'               : 'Fidelity F',
        'concurrence'            : 'Concurrence',
        'qber'                   : 'QBER',
        'transmittance'          : 'Transmittance',
        'distance_km'            : 'Distance (km)',
        'prop_latency_ms'        : 'Prop. Latency (ms)',
        'T_rtt_ms'               : 'RTT (ms)',
        'decoherence_time_ms'    : 'T₂ Coherence (ms)',
        'mean_attempts'          : 'Mean Attempts',
        'secret_key_rate_bps'    : 'SKR (bps)',
        'link_budget_db'         : 'Link Budget (dB)',
        'entanglement_gen_rate'  : 'Ent. Gen. Rate',
        'doppler_shift_hz'       : 'Doppler Shift (Hz)',
        'handover_prob'          : 'Handover Prob.',
        'effective_throughput_bps': 'Eff. Throughput',
    }

    datasets = []
    if len(bts_df):
        bts_cols = [c for c in bts_core if c in bts_df.columns]
        sub_bts  = bts_df[bts_cols].dropna()
        bts_cols = [c for c in bts_cols if sub_bts[c].std() > 0]
        sub_bts  = sub_bts[bts_cols]
        datasets.append((sub_bts, bts_cols, 'BTS Backbone Links'))

    mob_cols = [c for c in mob_core if c in mob_links.columns]
    sub_mob  = mob_links[mob_cols].dropna()
    mob_cols = [c for c in mob_cols if sub_mob[c].std() > 0]
    sub_mob  = sub_mob[mob_cols]
    datasets.append((sub_mob, mob_cols, 'Mobile Links'))

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(13 * n_ds, max(10, 12)))
    if n_ds == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')

    cmap_div = plt.cm.RdBu_r

    for ax_i, (sub, cols, title) in enumerate(datasets):
        ax = axes[ax_i]
        ax_style(ax, f'({chr(65+ax_i)})  Pearson Correlation -- {title}')

        corr = sub.corr()
        n    = len(cols)

        im = ax.imshow(corr.values, cmap=cmap_div, vmin=-1, vmax=1,
                       aspect='auto', interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, aspect=35)
        cbar.set_label('Pearson r', fontsize=9, color=DARK)
        cbar.ax.tick_params(labelsize=8)

        tick_labels = [label_map.get(c, c) for c in cols]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=38, ha='right',
                           fontsize=8.5, color=DARK)
        ax.set_yticklabels(tick_labels, fontsize=8.5, color=DARK)

        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                fc  = 'white' if abs(val) > 0.55 else DARK
                star = '★' if (i != j and abs(val) > 0.7) else ''
                ax.text(j, i, f'{val:.2f}{star}',
                        ha='center', va='center',
                        fontsize=8.2, color=fc, fontweight='bold')

        ax.text(0.01, -0.18,
                '⚠ Log. negativity excluded: constant ~ 0 across all simulated\n'
                '  Werner-state links (undefined correlation with any variable).',
                transform=ax.transAxes, fontsize=7.5, color=GRAY,
                va='top', style='italic')

    fig.suptitle('Quantum Link Metric Correlation Matrices -- BTS & Mobile\n'
                 '(★ = |r| > 0.70  |  negativity excluded -- constant for Werner states)',
                 fontsize=12, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_fig(fig, "fig15b_stats_correlation.png")


def _fig_stats_bts_nodes(bts_node_df):
    df = bts_node_df.sort_values('mean_link_fidelity', ascending=True)
    n  = len(df)

    fig, axes = plt.subplots(1, 3, figsize=(20, max(7, n*0.44)))
    fig.patch.set_facecolor('white')

    ax1 = axes[0]
    ax_style(ax1, '(A)  Mean Link Fidelity per BTS Node')
    bar_cols = [cmap_fid(f) for f in df['mean_link_fidelity']]
    hb = ax1.barh(range(n), df['mean_link_fidelity'],
                  color=bar_cols, edgecolor=DARK, lw=0.7, alpha=0.88,
                  height=0.62)
    ax1.errorbar(df['mean_link_fidelity'],
                 range(n),
                 xerr=df['mean_link_fidelity'] - df['min_link_fidelity'],
                 fmt='none', ecolor=DARK, lw=1.2, capsize=3, zorder=5)
    for bar, row in zip(hb, df.itertuples()):
        ax1.text(row.mean_link_fidelity + 0.005,
                 bar.get_y() + bar.get_height()/2,
                 f'{row.mean_link_fidelity:.3f}',
                 va='center', fontsize=7.5, color=DARK, fontweight='bold')
    ax1.axvline(0.5, color=RED,   ls=':', lw=1.5, label='Ent. threshold')
    ax1.axvline(2/3, color=GREEN, ls=':', lw=1.5, label='QKD threshold')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(df['bts_name'], fontsize=8, color=DARK)
    ax1.set_xlabel('Mean Link Fidelity F(Φ⁺)', fontsize=10, color=DARK)
    ax1.legend(fontsize=8, framealpha=0.92, edgecolor='#CCC', loc='lower right')
    ax1.set_xlim(0, 1.05)

    ax2 = axes[1]
    ax_style(ax2, '(B)  Node Degree  (BTS-to-BTS links)')
    type_col_map = {'hub': BLUE, 'relay': TEAL, 'node': ORANGE}
    deg_cols = [type_col_map.get(t, GRAY) for t in df['type']]
    hb2 = ax2.barh(range(n), df['degree'],
                   color=deg_cols, edgecolor=DARK, lw=0.7, alpha=0.85,
                   height=0.62)
    for bar, row in zip(hb2, df.itertuples()):
        ax2.text(row.degree + 0.1, bar.get_y() + bar.get_height()/2,
                 str(int(row.degree)), va='center', fontsize=8, color=DARK,
                 fontweight='bold')
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(df['bts_name'], fontsize=8, color=DARK)
    ax2.set_xlabel('Degree (# BTS neighbours)', fontsize=10, color=DARK)
    type_leg = [mpatches.Patch(color=BLUE,   alpha=0.85, label='Hub'),
                mpatches.Patch(color=TEAL,   alpha=0.85, label='Relay'),
                mpatches.Patch(color=ORANGE, alpha=0.85, label='Node')]
    ax2.legend(handles=type_leg, fontsize=8, framealpha=0.92,
               edgecolor='#CCC', loc='lower right')

    ax3 = axes[2]
    ax_style(ax3, '(C)  Total Reach  (sum of all link distances)')
    hb3 = ax3.barh(range(n), df['total_reach_km'],
                   color=PURPLE, edgecolor=DARK, lw=0.7, alpha=0.75,
                   height=0.62)
    for bar, row in zip(hb3, df.itertuples()):
        ax3.text(row.total_reach_km + 0.3,
                 bar.get_y() + bar.get_height()/2,
                 f'{row.total_reach_km:.1f} km',
                 va='center', fontsize=7.5, color=DARK, fontweight='bold')
    ax3.set_yticks(range(n))
    ax3.set_yticklabels(df['bts_name'], fontsize=8, color=DARK)
    ax3.set_xlabel('Total reach (km)', fontsize=10, color=DARK)

    fig.suptitle('BTS Node-Level Statistics -- Fidelity, Degree, Reach',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    save_fig(fig, "fig15c_stats_bts_nodes.png")


def _fig_stats_routing_detail(routing_table):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    ax = axes_flat[0]
    ax_style(ax, '(A)  E2E Fidelity Distribution  (all Dijkstra routes)')
    vals = routing_table['e2e_fidelity']
    ax.hist(vals, bins=28, color=BLUE, alpha=0.82, edgecolor='white', zorder=3)
    ax.axvline(vals.mean(),   color=DARK,  ls='--', lw=2.0,
               label=f'μ = {vals.mean():.4f}')
    ax.axvline(vals.median(), color=GRAY,  ls=':',  lw=1.8,
               label=f'Median = {vals.median():.4f}')
    ax.axvline(0.5,           color=RED,   ls=':', lw=1.8,
               label='Entanglement threshold')
    ax.axvline(2/3,           color=GREEN, ls=':', lw=1.8,
               label='QKD threshold')
    ax.set_xlabel('E2E Fidelity F(Φ⁺)', fontsize=10, color=DARK)
    ax.set_ylabel('Route count', fontsize=10, color=DARK)
    ax.legend(fontsize=8.5, framealpha=0.92, edgecolor='#CCC')

    ax = axes_flat[1]
    ax_style(ax, '(B)  Hop Count Distribution')
    hop_counts = routing_table['hops'].value_counts().sort_index()
    ax.bar(hop_counts.index, hop_counts.values,
           color=[cmap_fid(1 - 0.15*h) for h in hop_counts.index],
           edgecolor=DARK, lw=0.8, alpha=0.85, width=0.6)
    for hops, cnt in hop_counts.items():
        ax.text(hops, cnt + 0.5, str(cnt), ha='center', fontsize=9,
                fontweight='bold', color=DARK)
    ax.set_xlabel('Number of hops', fontsize=10, color=DARK)
    ax.set_ylabel('Number of routes', fontsize=10, color=DARK)
    ax.set_xticks(hop_counts.index)
    ax2_twin = ax.twiny()
    ax2_twin.set_xlim(ax.get_xlim())
    ax2_twin.set_xticks(hop_counts.index)
    ax2_twin.set_xticklabels([f'{h-1} swap{"s" if h-1!=1 else ""}'
                               for h in hop_counts.index],
                              fontsize=8, color=PURPLE)

    ax = axes_flat[2]
    ax_style(ax, '(C)  Routing Gain ΔF Distribution')
    gain = routing_table['routing_gain']
    ax.hist(gain[gain > 0], bins=20, color=GREEN, alpha=0.80,
            edgecolor='white', label='Positive gain (Dijkstra better)', zorder=3)
    ax.hist(gain[gain == 0], bins=5, color=BLUE, alpha=0.80,
            edgecolor='white', label='Zero gain (same path)', zorder=3)
    ax.hist(gain[gain < 0], bins=5, color=RED, alpha=0.80,
            edgecolor='white', label='Negative gain (degenerate)', zorder=3)
    ax.axvline(0, color=DARK, lw=1.5, ls='--')
    ax.set_xlabel('Routing Gain ΔF', fontsize=10, color=DARK)
    ax.set_ylabel('Route count', fontsize=10, color=DARK)
    ax.legend(fontsize=8.5, framealpha=0.92, edgecolor='#CCC')

    ax = axes_flat[3]
    ax_style(ax, '(D)  Path Distance by Hop Count')
    hop_groups = [routing_table[routing_table['hops']==h]['total_dist_km'].values
                  for h in sorted(routing_table['hops'].unique())]
    hop_labels  = [f'{h} hop{"s" if h>1 else ""}' for h in
                   sorted(routing_table['hops'].unique())]
    if hop_groups:
        vp = ax.violinplot(hop_groups, positions=range(len(hop_groups)),
                           showmedians=True, widths=0.65)
        for pc in vp['bodies']:
            pc.set_facecolor(PURPLE); pc.set_alpha(0.45)
        vp['cmedians'].set_color(DARK); vp['cmedians'].set_linewidth(2.5)
        for i, (vals_v, col_v) in enumerate(
                zip(hop_groups, [cmap_fid(1-0.15*i) for i in range(len(hop_groups))])):
            jitter = RNG.uniform(-0.12, 0.12, len(vals_v))
            ax.scatter(i + jitter, vals_v, s=18, c=col_v, alpha=0.5,
                       edgecolors='none', zorder=4)
    ax.set_xticks(range(len(hop_labels)))
    ax.set_xticklabels(hop_labels, fontsize=9, color=DARK)
    ax.set_xlabel('Hop count', fontsize=10, color=DARK)
    ax.set_ylabel('Total path distance (km)', fontsize=10, color=DARK)

    fig.suptitle('Extended Routing Statistics -- Dijkstra Entanglement-Swap Router',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, "fig15d_stats_routing_detail.png")
