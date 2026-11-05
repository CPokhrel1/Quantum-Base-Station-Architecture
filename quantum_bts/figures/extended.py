import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..channel import transmittance
from ..config import COHERENCE_BASE_MS, MAX_LINK_KM, MAX_MOBILE_KM, RNG
from .style import (
    ax_style, BLUE, DARK, GRAY, GREEN, MOB_TYPE_COLORS, MOB_TYPE_MARKERS, ORANGE, PINK,
    PURPLE, RED, save_fig, TEAL,
)
from ..link_metrics import (
    _doppler_shift, _handover_prob, _link_budget_db, _total_latency_ms,
)


def fig_latency_overview(mob_links, bts_link_df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    PHASE_COLORS = {
        'prop_latency_ms'        : '#1A6DB5',
        'herald_latency_ms'      : '#0E9E8E',
        'bsm_latency_ms'         : '#6A3FA8',
        'announce_latency_ms'    : '#E07B1A',
        'feedforward_latency_ms' : '#D63A2F',
    }
    PHASE_LABELS = {
        'prop_latency_ms'        : 'Ph1 Photon propagation  (d/c)',
        'herald_latency_ms'      : 'Ph2 Herald return       (d/c, classical)',
        'bsm_latency_ms'         : 'Ph3 BSM hardware        (~1 µs, fixed)',
        'announce_latency_ms'    : 'Ph4 Classical processing (~2 µs, fixed)',
        'feedforward_latency_ms' : 'Ph5 Feed-forward gate   (~5 µs, fixed)',
    }
    phase_cols = list(PHASE_COLORS.keys())

    ax = axes_flat[0]
    ax_style(ax, '(A)  Mean Single-Attempt RTT by Phase -- BTS vs Mobile\n'
             '(T_RTT = Ph1+Ph2+Ph3+Ph4; delivery latency T_del = T_RTT/P_s + Ph5)')

    categories = ['BTS backbone', 'Mobile -- all', 'Mobile -- ped.',
                  'Mobile -- vehicle', 'Mobile -- IoT']
    subsets = [
        bts_link_df if len(bts_link_df) else pd.DataFrame(),
        mob_links,
        mob_links[mob_links['mob_type'] == 'pedestrian'],
        mob_links[mob_links['mob_type'] == 'vehicle'],
        mob_links[mob_links['mob_type'] == 'IoT'],
    ]
    x = np.arange(len(categories))
    bottoms = np.zeros(len(categories))
    for pc in phase_cols:
        means = []
        for sub in subsets:
            means.append(sub[pc].mean() * 1e3 if (len(sub) and pc in sub.columns) else 0)
        means = np.array(means)
        ax.bar(x, means, bottom=bottoms,
               color=PHASE_COLORS[pc], alpha=0.88, label=PHASE_LABELS[pc],
               edgecolor='white', lw=0.6, width=0.6)
        bottoms += means

    for i, sub in enumerate(subsets):
        rtt = sub['T_rtt_ms'].mean() * 1e3 if (len(sub) and 'T_rtt_ms' in sub.columns) else 0
        ax.text(i, bottoms[i] + 0.2, f'T_RTT\n{rtt:.1f} µs',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, color=DARK)
    ax.set_ylabel('Single-Attempt RTT (µs)', fontsize=10, color=DARK)
    ax.legend(fontsize=7.8, framealpha=0.95, edgecolor='#CCC',
              loc='upper right', title='Protocol Phase', title_fontsize=8)

    ax = axes_flat[1]
    ax_style(ax, '(B)  Pair Delivery Latency T_del vs Distance  (log scale)\n'
             'T_del = T_RTT/P_s + delta_ff  -- dominated by probabilistic attempts')
    ax.set_yscale('log')
    if len(bts_link_df):
        ax.scatter(bts_link_df['distance_km'], bts_link_df['total_latency_ms'] / 1e3,
                   c=BLUE, s=90, alpha=0.85, edgecolors=DARK, lw=0.7,
                   marker='o', zorder=5, label='BTS backbone')
    for mtype, grp in mob_links.groupby('mob_type'):
        ax.scatter(grp['distance_km'], grp['total_latency_ms'] / 1e3,
                   c=MOB_TYPE_COLORS[mtype], marker=MOB_TYPE_MARKERS[mtype],
                   s=35, alpha=0.72, edgecolors='white', lw=0.4,
                   zorder=4, label=f'Mobile -- {mtype}')
    ds_th = np.linspace(0.05, MAX_MOBILE_KM, 200)
    T_del_th = [_total_latency_ms(d, transmittance(d)) / 1e3 for d in ds_th]
    ax.plot(ds_th, T_del_th, color=DARK, lw=2.5, ls='--', zorder=6,
            label='Theory: T_RTT(d)/P_s(d) + delta_ff')
    ax.axhline(0.1,  color=GREEN,  ls=':', lw=1.4, label='100 ms threshold')
    ax.axhline(1.0,  color=ORANGE, ls=':', lw=1.4, label='1 s threshold')
    ax.axhline(10.0, color=RED,    ls=':', lw=1.2, label='10 s threshold')
    ax.set_xlabel('Distance (km)', fontsize=10, color=DARK)
    ax.set_ylabel('Delivery Latency T_del (s)', fontsize=10, color=DARK)
    ax.legend(fontsize=8.0, framealpha=0.95, edgecolor='#CCC', loc='upper left')

    ax = axes_flat[2]
    ax.set_facecolor('white')
    ax_style(ax, '(C)  RTT Phase Composition -- Mobile Links (mean)\n'
             'Note: delivery latency T_del ≫ T_RTT due to 1/P_s repetitions')
    pie_vals = [mob_links[pc].mean() for pc in phase_cols]
    pie_labels = [PHASE_LABELS[pc].split('(')[0].strip() for pc in phase_cols]
    pie_colors = list(PHASE_COLORS.values())
    wedges, texts, autotexts = ax.pie(
        pie_vals, labels=pie_labels,
        colors=pie_colors, autopct='%1.1f%%',
        wedgeprops=dict(edgecolor='white', lw=2),
        textprops=dict(fontsize=8.0, color=DARK),
        startangle=90, pctdistance=0.75)
    for at in autotexts:
        at.set_fontsize(8.5); at.set_fontweight('bold')
    rtt_mean   = mob_links['T_rtt_ms'].mean()
    T_del_mean = mob_links['total_latency_ms'].mean()
    P_s_mean   = mob_links['success_prob'].mean()
    ax.text(0, -1.52,
            f'Mean T_RTT = {rtt_mean*1e3:.2f} µs  |  Mean P_s = {P_s_mean:.2e}\n'
            f'Mean T_del = {T_del_mean/1e3:.2f} s  (= T_RTT/P_s + delta_ff)',
            ha='center', fontsize=8.5, color=DARK,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#CBD5E1'))

    ax = axes_flat[3]
    ax_style(ax, '(D)  T₂ Coherence Time vs Single-Attempt RTT  (memory feasibility)\n'
             'T₂ must exceed T_RTT to store entanglement between attempts')
    if len(bts_link_df):
        ratio_bts = bts_link_df['decoherence_time_ms'] / bts_link_df['T_rtt_ms']
        sc = ax.scatter(bts_link_df['T_rtt_ms'] * 1e3,
                        bts_link_df['decoherence_time_ms'],
                        c=ratio_bts.clip(0, 200), cmap='RdYlGn',
                        vmin=0, vmax=200, s=90, alpha=0.85,
                        edgecolors=DARK, lw=0.7, marker='o', zorder=5,
                        label='BTS backbone')
    ratio_mob = mob_links['decoherence_time_ms'] / mob_links['T_rtt_ms']
    sc2 = ax.scatter(mob_links['T_rtt_ms'] * 1e3,
                     mob_links['decoherence_time_ms'],
                     c=ratio_mob.clip(0, 200), cmap='RdYlGn',
                     vmin=0, vmax=200, s=35, alpha=0.75,
                     edgecolors='white', lw=0.3, marker='D', zorder=4,
                     label='Mobile links')
    x_rtt = mob_links['T_rtt_ms'].values * 1e3
    if len(bts_link_df):
        x_rtt = np.concatenate([x_rtt, bts_link_df['T_rtt_ms'].values * 1e3])
    lim_max = max(x_rtt.max(), 1.0) * 1.2
    xs_feas = np.linspace(0, lim_max, 100)
    ax.plot(xs_feas, xs_feas * 1e-3, color=RED, lw=2, ls='--',
            label='T₂ = T_RTT  (memory just survives 1 RTT)')
    ax.fill_between(xs_feas, xs_feas * 1e-3, COHERENCE_BASE_MS * 2,
                    color=GREEN, alpha=0.08, label='Feasible: T₂ > T_RTT')
    ax.fill_between(xs_feas, 0, xs_feas * 1e-3,
                    color=RED, alpha=0.08, label='Infeasible: T₂ < T_RTT')
    cbar2 = fig.colorbar(sc2, ax=ax, fraction=0.04, pad=0.01)
    cbar2.set_label('T₂ / T_RTT  (ratio)', fontsize=9, color=DARK)
    ax.set_xlabel('Single-Attempt RTT  (µs)', fontsize=10, color=DARK)
    ax.set_ylabel('Memory Coherence T₂  (ms)', fontsize=10, color=DARK)
    ax.legend(fontsize=7.8, framealpha=0.95, edgecolor='#CCC', loc='upper right')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    fig.suptitle(
        'Quantum Entanglement Protocol Latency -- Physically Rigorous Model\n'
        'T_del = T_RTT/P_s + delta_ff  |  T_RTT = 2*d/c + delta_BSM + delta_class  '
        '|  P_s = p_emit*eta_c*eta_det^2  |  Memory: NV-centre T₂ (Hahn echo)',
        fontsize=12, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return save_fig(fig, "fig16_latency_overview.png")


def fig_purified_skr_comparison(mob_links, bts_link_df):
    has_bts = (bts_link_df is not None and not bts_link_df.empty
               and 'purified_fidelity' in bts_link_df.columns)
    has_mob = (mob_links is not None and not mob_links.empty
               and 'purified_fidelity' in mob_links.columns)
    if not has_bts and not has_mob:
        print("  [!] No single-edge purification data available, skipping fig20.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.patch.set_facecolor('white')

    bts_c, mob_c = BLUE, PURPLE

    ax = axes[0, 0]
    ax_style(ax, '(A)  Raw vs. Purified SKR  (single-hop links)')
    all_skr = []
    if has_bts:
        ax.scatter(bts_link_df['skr_raw_bps'], bts_link_df['skr_purified_bps'],
                  s=50, c=bts_c, edgecolor=DARK, lw=0.5, alpha=0.75,
                  label=f'BTS-BTS (n={len(bts_link_df)})', zorder=3)
        all_skr += [bts_link_df['skr_raw_bps'].max(), bts_link_df['skr_purified_bps'].max()]
    if has_mob:
        ax.scatter(mob_links['skr_raw_bps'], mob_links['skr_purified_bps'],
                  s=30, c=mob_c, edgecolor=DARK, lw=0.4, alpha=0.6,
                  label=f'Mobile-BTS (n={len(mob_links)})', zorder=3)
        all_skr += [mob_links['skr_raw_bps'].max(), mob_links['skr_purified_bps'].max()]
    lim = max(all_skr) * 1.05 if all_skr else 1.0
    ax.plot([0, lim], [0, lim], color=DARK, ls='--', lw=1.2, alpha=0.6, zorder=2,
           label='No change (y = x)')
    ax.set_xlabel('Raw SKR (bps)', fontsize=10, color=DARK)
    ax.set_ylabel('Purified SKR (bps)', fontsize=10, color=DARK)
    ax.legend(fontsize=8, framealpha=0.95, edgecolor='#CCC', loc='upper left')

    ax = axes[0, 1]
    ax_style(ax, '(B)  Links Rescued from SKR = 0 by Purification')
    cats, rescued, still_dead, always_alive = [], [], [], []
    for label, df in [('BTS-BTS', bts_link_df if has_bts else None),
                      ('Mobile-BTS', mob_links if has_mob else None)]:
        if df is None or df.empty:
            continue
        cats.append(label)
        was_dead = df['skr_raw_bps'] <= 0
        now_alive = df['skr_purified_bps'] > 0
        rescued.append(int((was_dead & now_alive).sum()))
        still_dead.append(int((was_dead & ~now_alive).sum()))
        always_alive.append(int((~was_dead).sum()))
    x = np.arange(len(cats))
    if cats:
        ax.bar(x, always_alive, color=GREEN, label='Already QKD-capable', zorder=3)
        ax.bar(x, rescued, bottom=always_alive, color=ORANGE,
              label='Rescued by purification', zorder=3)
        bottom2 = [a + r for a, r in zip(always_alive, rescued)]
        ax.bar(x, still_dead, bottom=bottom2, color=RED,
              label='Still SKR = 0', zorder=3)
        for i, (a, r, s) in enumerate(zip(always_alive, rescued, still_dead)):
            total = a + r + s
            if total > 0:
                ax.text(i, total + total*0.02, f'n={total}', ha='center',
                       fontsize=8.5, fontweight='bold', color=DARK)
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9.5)
    ax.set_ylabel('Number of Links', fontsize=10, color=DARK)
    ax.legend(fontsize=8, framealpha=0.95, edgecolor='#CCC', loc='upper right')

    ax = axes[1, 0]
    ax_style(ax, '(C)  QBER Reduction from Purification')
    if has_bts:
        ax.scatter(bts_link_df['distance_km'], bts_link_df['qber'],
                  s=22, c=bts_c, marker='o', alpha=0.45, zorder=2,
                  label='BTS raw')
        ax.scatter(bts_link_df['distance_km'], bts_link_df['purified_qber'],
                  s=22, c=bts_c, marker='^', alpha=0.85, zorder=3,
                  label='BTS purified')
    if has_mob:
        ax.scatter(mob_links['distance_km'], mob_links['qber'],
                  s=14, c=mob_c, marker='o', alpha=0.35, zorder=2,
                  label='Mobile raw')
        ax.scatter(mob_links['distance_km'], mob_links['purified_qber'],
                  s=14, c=mob_c, marker='^', alpha=0.7, zorder=3,
                  label='Mobile purified')
    ax.axhline(0.11, color=RED, ls=':', lw=1.5, label='BB84 limit (11%)')
    ax.set_xlabel('Distance (km)', fontsize=10, color=DARK)
    ax.set_ylabel('QBER', fontsize=10, color=DARK)
    ax.legend(fontsize=7.5, framealpha=0.95, edgecolor='#CCC', ncol=2)

    ax = axes[1, 1]
    ax_style(ax, '(D)  Aggregate Network SKR Budget')
    budget_labels, raw_totals, pur_totals = [], [], []
    if has_bts:
        budget_labels.append('BTS-BTS')
        raw_totals.append(bts_link_df['skr_raw_bps'].sum())
        pur_totals.append(bts_link_df['skr_purified_bps'].sum())
    if has_mob:
        budget_labels.append('Mobile-BTS')
        raw_totals.append(mob_links['skr_raw_bps'].sum())
        pur_totals.append(mob_links['skr_purified_bps'].sum())
    if budget_labels:
        xb = np.arange(len(budget_labels))
        w = 0.35
        b1 = ax.bar(xb - w/2, raw_totals, w, color=GRAY, edgecolor=DARK,
                   lw=0.7, alpha=0.85, label='Raw (no purification)', zorder=3)
        b2 = ax.bar(xb + w/2, pur_totals, w, color=TEAL, edgecolor=DARK,
                   lw=0.7, alpha=0.85, label='Purified', zorder=3)
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                       f'{b.get_height():,.0f}', ha='center', va='bottom',
                       fontsize=8, color=DARK)
        ax.set_xticks(xb); ax.set_xticklabels(budget_labels, fontsize=9.5)
    ax.set_ylabel('Sum of SKR across all links (bps)', fontsize=10, color=DARK)
    ax.legend(fontsize=8.5, framealpha=0.95, edgecolor='#CCC')

    fig.suptitle('Single-Hop Purified Secret Key Rate: BTS-BTS & Mobile-BTS Links\n'
                 '(purify_link() / add_single_edge_purification() -- distinct from the\n'
                 'multi-hop path purification shown in fig14e_purification_comparison.png)',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig20_purified_skr_comparison.png")


def fig_skr_analysis(mob_links, bts_link_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    ax = axes_flat[0]
    ax_style(ax, '(A)  Secret-Key Rate Distribution')
    if len(bts_link_df):
        bts_skr = bts_link_df['secret_key_rate_bps'].dropna()
        ax.hist(bts_skr/1e3, bins=16, color=BLUE, alpha=0.75, edgecolor='white',
                label=f'BTS  (μ={bts_skr.mean()/1e3:.2f} kbps)', zorder=3)
    mob_skr = mob_links['secret_key_rate_bps'].dropna()
    ax.hist(mob_skr/1e3, bins=22, color=PINK, alpha=0.65, edgecolor='white',
            label=f'Mobile  (μ={mob_skr.mean()/1e3:.2f} kbps)', zorder=4)
    ax.set_xlabel('Secret-Key Rate (kbps)', fontsize=10, color=DARK)
    ax.set_ylabel('Count', fontsize=10, color=DARK)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC')

    ax = axes_flat[1]
    ax_style(ax, '(B)  SKR vs Distance  (colour = QBER)')
    sc = ax.scatter(mob_links['distance_km'], mob_links['secret_key_rate_bps']/1e3,
                    c=mob_links['qber'], cmap='RdYlGn_r', vmin=0, vmax=0.15,
                    s=25, alpha=0.70, edgecolors='none', zorder=4, label='Mobile')
    if len(bts_link_df):
        ax.scatter(bts_link_df['distance_km'], bts_link_df['secret_key_rate_bps']/1e3,
                   c=BLUE, s=100, alpha=0.85, edgecolors=DARK, lw=0.8,
                   marker='*', zorder=5, label='BTS')
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_label('QBER', fontsize=9, color=DARK)
    cbar.ax.axhline(0.11, color=RED, lw=1.5, ls='--')
    ax.set_xlabel('Distance (km)', fontsize=10, color=DARK)
    ax.set_ylabel('SKR (kbps)', fontsize=10, color=DARK)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC')

    ax = axes_flat[2]
    ax_style(ax, '(C)  SKR by Mobile Type  (violin + strip)')
    mob_types = sorted(mob_links['mob_type'].unique())
    data_v = [mob_links[mob_links['mob_type']==mt]['secret_key_rate_bps'].values/1e3
              for mt in mob_types]
    vp = ax.violinplot(data_v, positions=range(len(mob_types)),
                       showmedians=True, widths=0.65)
    for pc, mt in zip(vp['bodies'], mob_types):
        pc.set_facecolor(MOB_TYPE_COLORS[mt]); pc.set_alpha(0.5)
    vp['cmedians'].set_color(DARK); vp['cmedians'].set_linewidth(2.5)
    for i, (vals_v, mt) in enumerate(zip(data_v, mob_types)):
        jit = RNG.uniform(-0.1, 0.1, len(vals_v))
        ax.scatter(i+jit, vals_v, s=10, c=MOB_TYPE_COLORS[mt],
                   alpha=0.4, edgecolors='none', zorder=4)
        ax.text(i, np.mean(vals_v)*1.05,
                f'μ={np.mean(vals_v):.2f}', ha='center',
                fontsize=8.5, fontweight='bold', color=DARK)
    ax.set_xticks(range(len(mob_types)))
    ax.set_xticklabels(mob_types, fontsize=10, color=DARK)
    ax.set_ylabel('SKR (kbps)', fontsize=10, color=DARK)

    ax = axes_flat[3]
    ax_style(ax, '(D)  Effective Throughput vs Speed  (SKR * (1−P_HO))')
    for mtype, grp in mob_links.groupby('mob_type'):
        ax.scatter(grp['speed_kmh'], grp['effective_throughput_bps']/1e3,
                   c=MOB_TYPE_COLORS[mtype], marker=MOB_TYPE_MARKERS[mtype],
                   s=35, alpha=0.7, edgecolors='white', lw=0.4, zorder=4, label=mtype)
        coef = np.polyfit(grp['speed_kmh'], grp['effective_throughput_bps']/1e3, 1)
        xs = np.linspace(0, grp['speed_kmh'].max(), 80)
        ax.plot(xs, np.polyval(coef, xs), color=MOB_TYPE_COLORS[mtype], lw=2, ls='--')
    ax.set_xlabel('Speed (km/h)', fontsize=10, color=DARK)
    ax.set_ylabel('Effective Throughput (kbps)', fontsize=10, color=DARK)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC',
              title='Mobile type + trend', title_fontsize=8)

    fig.suptitle('Secret-Key Rate & Effective Throughput  |  BB84 Shor-Preskill Bound',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, "fig17_skr_analysis.png")


def fig_link_budget(mob_links, bts_link_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    ax = axes_flat[0]
    ax_style(ax, '(A)  Optical Link Budget vs Distance')
    if len(bts_link_df):
        ax.scatter(bts_link_df['distance_km'], bts_link_df['link_budget_db'],
                   c=BLUE, s=90, alpha=0.85, edgecolors=DARK, lw=0.7,
                   marker='o', zorder=5, label='BTS backbone')
    sc = ax.scatter(mob_links['distance_km'], mob_links['link_budget_db'],
                    c=mob_links['speed_kmh'], cmap='plasma', vmin=0, vmax=80,
                    s=25, alpha=0.65, edgecolors='none', marker='D', zorder=4, label='Mobile')
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01).set_label('Speed (km/h)', fontsize=9, color=DARK)
    ds_lb = np.linspace(0.05, MAX_LINK_KM, 200)
    ax.plot(ds_lb, [_link_budget_db(transmittance(d)) for d in ds_lb],
            color=DARK, lw=2, ls='--', label='Theoretical', zorder=6)
    ax.set_xlabel('Distance (km)', fontsize=10, color=DARK)
    ax.set_ylabel('Link Budget Loss (dB)', fontsize=10, color=DARK)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC')

    ax = axes_flat[1]
    ax_style(ax, '(B)  Doppler Frequency Shift vs Speed')
    for mtype, grp in mob_links.groupby('mob_type'):
        ax.scatter(grp['speed_kmh'], grp['doppler_shift_hz']/1e6,
                   c=MOB_TYPE_COLORS[mtype], marker=MOB_TYPE_MARKERS[mtype],
                   s=40, alpha=0.75, edgecolors='white', lw=0.4, zorder=4, label=mtype)
    xs_d = np.linspace(0, 80, 100)
    ax.plot(xs_d, [_doppler_shift(v)/1e6 for v in xs_d],
            color=DARK, lw=2.5, ls='--', label='Theory: f_D = v/λ', zorder=6)
    ax.set_xlabel('Speed (km/h)', fontsize=10, color=DARK)
    ax.set_ylabel('Doppler Shift (MHz)', fontsize=10, color=DARK)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC',
              title='Mobile type', title_fontsize=8)

    ax = axes_flat[2]
    ax_style(ax, '(C)  Beam-Pointing Loss vs Speed  (colour = distance)')
    sc2 = ax.scatter(mob_links['speed_kmh'], mob_links['pointing_loss_db'],
                     c=mob_links['distance_km'], cmap='viridis',
                     s=30, alpha=0.70, edgecolors='none', zorder=4)
    fig.colorbar(sc2, ax=ax, fraction=0.04, pad=0.01).set_label('Distance (km)', fontsize=9, color=DARK)
    ax.set_xlabel('Speed (km/h)', fontsize=10, color=DARK)
    ax.set_ylabel('Pointing Loss (dB)', fontsize=10, color=DARK)
    ax.set_yscale('symlog', linthresh=0.001)

    ax = axes_flat[3]
    ax_style(ax, '(D)  Handover Probability vs Speed')
    speed_bins = np.linspace(0, 80, 10)
    bin_centres = 0.5*(speed_bins[:-1]+speed_bins[1:])
    bw = (speed_bins[1]-speed_bins[0])*0.28
    for i_mt, mtype in enumerate(['pedestrian','vehicle','IoT']):
        grp = mob_links[mob_links['mob_type']==mtype]
        means = []
        for b in range(len(speed_bins)-1):
            sub = grp[(grp['speed_kmh']>=speed_bins[b]) & (grp['speed_kmh']<speed_bins[b+1])]['handover_prob']
            means.append(sub.mean() if len(sub) else 0)
        ax.bar(bin_centres + (i_mt-1)*bw, means, width=bw,
               color=MOB_TYPE_COLORS[mtype], alpha=0.82,
               edgecolor=DARK, lw=0.6, label=mtype)
    xs_h = np.linspace(0, 80, 200)
    mean_T_rtt = mob_links['T_rtt_ms'].mean() if 'T_rtt_ms' in mob_links.columns else 0.02
    ax.plot(xs_h, [_handover_prob(v, mean_T_rtt) for v in xs_h],
            color=DARK, lw=2, ls='--',
            label=f'Theory P_HO  (T_RTT = {mean_T_rtt*1e3:.1f} µs mean)', zorder=6)
    ax.set_xlabel('Speed (km/h)', fontsize=10, color=DARK)
    ax.set_ylabel('Handover Probability', fontsize=10, color=DARK)
    ho_max = mob_links['handover_prob'].max() if len(mob_links) else 0.01
    ax.set_ylim(0, max(ho_max * 1.25, 1e-6))
    ax.legend(fontsize=8.5, framealpha=0.95, edgecolor='#CCC',
              ncol=2, title='Mobile type / theory', title_fontsize=8)

    fig.suptitle('Link Budget, Doppler Shift & Mobile Impairments\n'
                 'Beam Pointing Loss * Handover Probability  |  Kathmandu Network',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, "fig18_link_budget.png")


def fig_extended_metrics_dashboard(mob_links, bts_link_df):
    metrics_info = [
        ('T_rtt_ms',                'Single-Attempt RTT (µs)',       BLUE,   False, 1e3  ),
        ('total_latency_ms',        'Delivery Latency T_del (s)',    TEAL,   True,  1e-3 ),
        ('decoherence_time_ms',     'Memory T₂ Coherence (ms)',      GREEN,  False, 1.0  ),
        ('secret_key_rate_bps',     'Secret-Key Rate (bps)',         PURPLE, True,  1.0  ),
        ('entanglement_gen_rate',   'Entangled Pairs/s',             ORANGE, True,  1.0  ),
        ('effective_throughput_bps','Effective Throughput (bps)',    PINK,   True,  1.0  ),
        ('link_budget_db',          'Link Budget Loss (dB)',         RED,    False, 1.0  ),
        ('doppler_shift_hz',        'Doppler Shift (Hz)',            GRAY,   True,  1.0  ),
        ('handover_prob',           'Handover Probability',          ORANGE, False, 1.0  ),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    for ax_i, (col, label, color, log_scale, scale) in enumerate(metrics_info):
        ax = axes_flat[ax_i]
        ax_style(ax, f'({chr(65+ax_i)})  {label}')
        vals = mob_links[col].dropna() * scale
        if log_scale and (vals > 0).all():
            ax.set_xscale('log')
        ax.hist(vals, bins=22, color=color, alpha=0.80, edgecolor='white', lw=0.4, zorder=3)
        mu, sd = vals.mean(), vals.std()
        ax.axvline(mu, color=DARK, ls='--', lw=2.0, label=f'μ = {mu:.3g}')
        ax.axvline(mu - sd, color=GRAY, ls=':', lw=1.2)
        ax.axvline(mu + sd, color=GRAY, ls=':', lw=1.2)
        ax.set_xlabel(label, fontsize=8.5, color=DARK)
        ax.set_ylabel('Count', fontsize=8.5, color=DARK)
        ax.legend(fontsize=7.5, framealpha=0.92, edgecolor='#CCC')
        cv = sd / mu * 100 if mu else 0
        ax.text(0.97, 0.92, f'CV={cv:.1f}%\nσ={sd:.3g}',
                transform=ax.transAxes, fontsize=7.5, ha='right', va='top', color=DARK,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCC', alpha=0.9))

    fig.suptitle('Extended Metrics Dashboard -- Mobile FSO Quantum Links\n'
                 'Latency (RTT in µs, T_del in s) * T₂ * SKR * Throughput * Doppler * Handover',
                 fontsize=13, fontweight='bold', color=DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return save_fig(fig, "fig19_extended_metrics_dashboard.png")
