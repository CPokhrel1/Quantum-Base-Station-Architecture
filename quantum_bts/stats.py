import numpy as np
import pandas as pd

from .config import N_MOBILE, OUT
from .figures.stats import (
    _fig_stats_bts_nodes, _fig_stats_correlation, _fig_stats_overview,
    _fig_stats_routing_detail,
)


def compute_all_stats(G, mob_links, mobile_df, routing_table=None):
    edges = list(G.edges(data=True))
    bts_rows = [
        {**{'source': u, 'destination': v, 'link_type': 'bts-bts'},
         **{k: round(v2, 5) if isinstance(v2, float) else v2
            for k, v2 in d.items()}}
        for u, v, d in edges
    ]
    bts_df = pd.DataFrame(bts_rows) if bts_rows else pd.DataFrame()

    bts_df.sort_values('distance_km').to_csv(
        f"{OUT}/stats_bts_links.csv", index=False)
    mob_links.sort_values('fidelity', ascending=False).to_csv(
        f"{OUT}/stats_mobile_links.csv", index=False)
    mobile_df.to_csv(f"{OUT}/stats_mobile_nodes.csv", index=False)

    SCI_COLS = {
        'mean_attempts', 'success_prob', 'throughput_eps',
        'entanglement_gen_rate', 'secret_key_rate_bps',
        'effective_throughput_bps', 'handover_prob',
        'doppler_shift_hz', 'pointing_loss_db',
    }
    UNIT_MAP = {
        'prop_latency_ms'       : 'prop_latency (ms)',
        'herald_latency_ms'     : 'herald_latency (ms)',
        'bsm_latency_ms'        : 'bsm_latency (ms)',
        'announce_latency_ms'   : 'announce_latency (ms)',
        'feedforward_latency_ms': 'feedforward_latency (ms)',
        'T_rtt_ms'              : 'T_RTT (ms)',
        'total_latency_ms'      : 'T_del (ms)',
        'decoherence_time_ms'   : 'T2_coherence (ms)',
        'distance_km'           : 'distance (km)',
        'link_budget_db'        : 'link_budget (dB)',
        'doppler_shift_hz'      : 'doppler_shift (Hz)',
        'pointing_loss_db'      : 'pointing_loss (dB)',
        'secret_key_rate_bps'   : 'SKR (bps)',
        'effective_throughput_bps': 'eff_throughput (bps)',
        'entanglement_gen_rate' : 'ent_gen_rate (pairs/s)',
        'throughput_eps'        : 'throughput (pairs/s)',
        'mean_attempts'         : 'mean_attempts (count)',
        'success_prob'          : 'success_prob (per attempt)',
        'memory_feasible'       : 'memory_feasible (0/1)',
        'handover_prob'         : 'handover_prob (per RTT)',
    }

    def fmt(col, val):
        if col in SCI_COLS:
            return f"{val:.4e}"
        if abs(val) >= 1000:
            return f"{val:.2f}"
        if abs(val) >= 1:
            return f"{val:.4f}"
        return f"{val:.6f}"

    def summary(df, cols):
        rows = []
        for c in cols:
            v = df[c].dropna()
            label = UNIT_MAP.get(c, c)
            rows.append({
                'Metric'  : label,
                'Count'   : len(v),
                'Mean'    : fmt(c, v.mean()),
                'Std'     : fmt(c, v.std()),
                'Min'     : fmt(c, v.min()),
                'P10'     : fmt(c, v.quantile(0.10)),
                'P25'     : fmt(c, v.quantile(0.25)),
                'Median'  : fmt(c, v.median()),
                'P75'     : fmt(c, v.quantile(0.75)),
                'P90'     : fmt(c, v.quantile(0.90)),
                'Max'     : fmt(c, v.max()),
                'Skewness': f"{v.skew():.3f}",
                'Kurtosis': f"{v.kurt():.3f}",
            })
        return pd.DataFrame(rows)

    met_cols = ['fidelity', 'concurrence', 'qber', 'negativity',
                'transmittance', 'distance_km',
                'prop_latency_ms', 'T_rtt_ms', 'total_latency_ms',
                'mean_attempts', 'success_prob',
                'decoherence_time_ms', 'memory_feasible',
                'secret_key_rate_bps',
                'entanglement_gen_rate', 'link_budget_db',
                'doppler_shift_hz', 'pointing_loss_db',
                'handover_prob', 'effective_throughput_bps']
    bts_met_cols = [c for c in met_cols if c in bts_df.columns] if len(bts_df) else []
    mob_met_cols = [c for c in met_cols if c in mob_links.columns]

    bts_stats = summary(bts_df,    bts_met_cols) if len(bts_df) and bts_met_cols else pd.DataFrame()
    mob_stats = summary(mob_links, mob_met_cols)

    per_type_base = [c for c in mob_met_cols if c in [
        'fidelity','concurrence','qber','negativity','transmittance','distance_km',
        'prop_latency_ms','T_rtt_ms','total_latency_ms','mean_attempts',
        'decoherence_time_ms','memory_feasible',
        'secret_key_rate_bps','effective_throughput_bps','link_budget_db',
        'doppler_shift_hz','handover_prob']]

    per_type_rows = []
    for mtype, grp in mob_links.groupby('mob_type'):
        for c in per_type_base:
            v = grp[c].dropna()
            label = UNIT_MAP.get(c, c)
            per_type_rows.append({
                'Type'  : mtype,
                'Metric': label,
                'Count' : len(v),
                'Mean'  : fmt(c, v.mean()),
                'Std'   : fmt(c, v.std()),
                'Min'   : fmt(c, v.min()),
                'Median': fmt(c, v.median()),
                'Max'   : fmt(c, v.max()),
                'IQR'   : fmt(c, v.quantile(0.75) - v.quantile(0.25)),
                'CV%'   : f"{v.std()/v.mean()*100:.1f}%" if v.mean() else 'N/A',
            })
    per_type_df = pd.DataFrame(per_type_rows)

    bts_node_rows = []
    for node, nd in G.nodes(data=True):
        nbrs = list(G.neighbors(node))
        fids  = [G[node][nb]['fidelity']     for nb in nbrs] if nbrs else [0]
        dists = [G[node][nb]['distance_km']  for nb in nbrs] if nbrs else [0]
        skrs  = [G[node][nb].get('secret_key_rate_bps', 0) for nb in nbrs] if nbrs else [0]
        lats  = [G[node][nb].get('total_latency_ms', 0)    for nb in nbrs] if nbrs else [0]
        bts_node_rows.append({
            'bts_name'                   : node,
            'type'                       : nd['ntype'],
            'degree'                     : len(nbrs),
            'n_cells'                    : nd.get('n_cells', 0),
            'mean_link_fidelity'         : round(np.mean(fids),  6),
            'min_link_fidelity'          : round(min(fids),      6),
            'max_link_fidelity'          : round(max(fids),      6),
            'mean_link_dist_km'          : round(np.mean(dists), 4),
            'total_reach_km'             : round(sum(dists),     4),
            'mean_skr_bps'               : round(np.mean(skrs),  4),
            'total_skr_capacity_bps'     : round(sum(skrs),      4),
            'mean_delivery_latency_s'    : round(np.mean(lats) / 1e3, 4),
        })
    bts_node_df = pd.DataFrame(bts_node_rows).sort_values(
        'mean_link_fidelity', ascending=False)

    best_per_mob = mob_links.groupby('mobile_id')['fidelity'].max()
    n_total = len(best_per_mob)
    n_qkd   = int((best_per_mob > 2/3).sum())
    n_ent   = int((best_per_mob > 0.5).sum())
    n_sub   = int(((best_per_mob > 0) & (best_per_mob <= 0.5)).sum())
    n_none  = int((best_per_mob == 0).sum())

    n_unreachable = N_MOBILE - n_total

    cover_df = pd.DataFrame([
        {'Category': 'QKD-ready  (F > 0.667)', 'Count': n_qkd,
         'Pct': f"{n_qkd/N_MOBILE*100:.1f}%"},
        {'Category': 'Entangled  (0.5 < F <= 0.667)', 'Count': n_ent - n_qkd,
         'Pct': f"{(n_ent-n_qkd)/N_MOBILE*100:.1f}%"},
        {'Category': 'Sub-threshold  (0 < F <= 0.5)', 'Count': n_sub,
         'Pct': f"{n_sub/N_MOBILE*100:.1f}%"},
        {'Category': 'No BTS in range', 'Count': n_unreachable,
         'Pct': f"{n_unreachable/N_MOBILE*100:.1f}%"},
    ])

    if routing_table is not None and len(routing_table):
        rt_stats = pd.DataFrame([{
            'Metric': 'E2E fidelity',
            'Mean': f"{routing_table['e2e_fidelity'].mean():.4f}",
            'Std':  f"{routing_table['e2e_fidelity'].std():.4f}",
            'Min':  f"{routing_table['e2e_fidelity'].min():.4f}",
            'Median': f"{routing_table['e2e_fidelity'].median():.4f}",
            'Max':  f"{routing_table['e2e_fidelity'].max():.4f}",
        }, {
            'Metric': 'Path hops',
            'Mean': f"{routing_table['hops'].mean():.2f}",
            'Std':  f"{routing_table['hops'].std():.2f}",
            'Min':  f"{routing_table['hops'].min()}",
            'Median': f"{routing_table['hops'].median():.1f}",
            'Max':  f"{routing_table['hops'].max()}",
        }, {
            'Metric': 'Total path dist (km)',
            'Mean': f"{routing_table['total_dist_km'].mean():.2f}",
            'Std':  f"{routing_table['total_dist_km'].std():.2f}",
            'Min':  f"{routing_table['total_dist_km'].min():.2f}",
            'Median': f"{routing_table['total_dist_km'].median():.2f}",
            'Max':  f"{routing_table['total_dist_km'].max():.2f}",
        }, {
            'Metric': 'Routing gain ΔF',
            'Mean': f"{routing_table['routing_gain'].mean():.5f}",
            'Std':  f"{routing_table['routing_gain'].std():.5f}",
            'Min':  f"{routing_table['routing_gain'].min():.5f}",
            'Median': f"{routing_table['routing_gain'].median():.5f}",
            'Max':  f"{routing_table['routing_gain'].max():.5f}",
        }])
        rt_stats.to_csv(f"{OUT}/stats_routing_summary.csv", index=False)
    else:
        rt_stats = None

    bts_stats.to_csv(f"{OUT}/stats_bts_summary.csv",         index=False)
    mob_stats.to_csv(f"{OUT}/stats_mobile_summary.csv",      index=False)
    per_type_df.to_csv(f"{OUT}/stats_mobile_per_type.csv",   index=False)
    bts_node_df.to_csv(f"{OUT}/stats_bts_nodes.csv",         index=False)
    cover_df.to_csv(f"{OUT}/stats_coverage.csv",             index=False)

    for title, df in [
        ("BTS LINK SUMMARY  (extended)",      bts_stats),
        ("MOBILE LINK SUMMARY  (extended)",   mob_stats),
        ("MOBILE STATS BY TYPE",              per_type_df),
        ("BTS NODE-LEVEL STATS",              bts_node_df),
        ("MOBILE COVERAGE BREAKDOWN",         cover_df),
    ]:
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
        if len(df): print(df.to_string(index=False))

    if rt_stats is not None:
        print("\n" + "="*80)
        print("  DIJKSTRA ROUTING SUMMARY STATS")
        print("="*80)
        print(rt_stats.to_string(index=False))
        n_qkd_rt  = routing_table['qkd_ready'].sum()
        n_ent_rt  = routing_table['entangled'].sum()
        n_tot_rt  = len(routing_table)
        print(f"\n  QKD-ready routes : {n_qkd_rt}/{n_tot_rt}  ({n_qkd_rt/n_tot_rt*100:.1f}%)")
        print(f"  Entangled routes : {n_ent_rt}/{n_tot_rt}  ({n_ent_rt/n_tot_rt*100:.1f}%)")

    _fig_stats_overview(bts_df, mob_links, bts_node_df, cover_df)
    _fig_stats_correlation(bts_df, mob_links)
    _fig_stats_bts_nodes(bts_node_df)
    if routing_table is not None and len(routing_table):
        _fig_stats_routing_detail(routing_table)

    return bts_stats, mob_stats, per_type_df
