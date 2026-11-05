import heapq
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from .config import OUT, PURIFICATION_F_LOW, QKD_FIDELITY_THRESH, SWAP_FIDELITY_THRESH
from .figures.routing import (
    fig_purification_comparison, fig_routing_all_paths, fig_routing_gain,
    fig_routing_heatmap, fig_routing_hop_profiles,
)
from .link_metrics import _e2e_throughput_eps, _secret_key_rate_bps
from .purification import purify_bbpssw, purify_path
from .quantum_metrics import qber_from_rho
from .swapping import (
    e2e_fidelity, real_e2e_fidelity_for_path, regenerate_hop_state, swap_bsm,
)


def _edge_log_cost(f: float) -> float:
    return -np.log(max(f, 1e-9))


def dijkstra_max_fidelity(G: nx.Graph, source: str, target: str, n_repeats: int = 20):
    INF = float('inf')
    dist  = {n: INF for n in G.nodes()}
    prev  = {n: None for n in G.nodes()}
    dist[source] = 0.0
    pq = [(0.0, source, [source])]

    best_path = None
    best_cost = INF

    while pq:
        cost, node, path = heapq.heappop(pq)
        if cost > best_cost:
            continue
        if node == target:
            if cost < best_cost:
                best_cost = cost
                best_path = path
            continue
        for nbr in G.neighbors(node):
            edge_f = G[node][nbr].get('fidelity', 0.0)
            new_cost = cost + _edge_log_cost(edge_f)
            if new_cost < dist[nbr]:
                dist[nbr] = new_cost
                heapq.heappush(pq, (new_cost, nbr, path + [nbr]))

    if best_path is None or len(best_path) < 2:
        return None, 0.0, [], 0.0, 0, 0.0, 0.0

    hop_fids       = [G[u][v]['fidelity'] for u, v in zip(best_path[:-1], best_path[1:])]
    e2e_F_analytic = e2e_fidelity(hop_fids)
    e2e_F, e2e_F_std, _ = real_e2e_fidelity_for_path(G, best_path, n_repeats=n_repeats)
    total_dist     = sum(G[u][v]['distance_km'] for u, v in zip(best_path[:-1], best_path[1:]))
    n_swaps        = len(best_path) - 2

    return best_path, e2e_F, hop_fids, total_dist, n_swaps, e2e_F_std, e2e_F_analytic


def run_routing_engine(G: nx.Graph, apply_purification: bool = True,
                        purify_n_repeats: int = 10):
    candidates = [n for n, d in G.nodes(data=True)
                  if d['ntype'] in ('hub', 'relay')]
    if len(candidates) < 2:
        candidates = list(G.nodes())

    pairs = [p for p in combinations(candidates, 2) if p[0] != p[1]]
    if apply_purification:
        print(f"    [routing] {len(pairs)} candidate pairs; purification "
              f"enabled (purify_n_repeats={purify_n_repeats}) -- this stage "
              f"may take a few minutes.")

    rows = []
    all_paths = []

    for pair_i, (src, dst) in enumerate(pairs):
        if not nx.has_path(G, src, dst):
            continue

        (path, e2e_F, hop_fids, total_dist, n_swaps,
         e2e_F_std, e2e_F_analytic) = dijkstra_max_fidelity(G, src, dst)
        if path is None:
            continue

        try:
            hop_path = nx.shortest_path(G, src, dst)
            hp_fids  = [G[u][v]['fidelity']
                        for u, v in zip(hop_path[:-1], hop_path[1:])]
            hp_e2e, hp_e2e_std, _ = real_e2e_fidelity_for_path(G, hop_path, n_repeats=20)
        except Exception:
            hop_path, hp_e2e = path, e2e_F

        direct_F = G[src][dst]['fidelity'] if G.has_edge(src, dst) else None

        row = {
            'source'          : src,
            'destination'     : dst,
            'path'            : ' -> '.join(path),
            'hops'            : len(path) - 1,
            'n_swaps'         : n_swaps,
            'e2e_fidelity'    : round(e2e_F, 5),
            'e2e_fidelity_std': round(e2e_F_std, 5),
            'e2e_fidelity_analytic': round(e2e_F_analytic, 5),
            'total_dist_km'   : round(total_dist, 3),
            'min_hop_fidelity': round(min(hop_fids), 5),
            'max_hop_fidelity': round(max(hop_fids), 5),
            'mean_hop_fidelity': round(np.mean(hop_fids), 5),
            'direct_fidelity' : round(direct_F, 5) if direct_F else None,
            'improvement_over_direct': (
                round(e2e_F - direct_F, 5) if direct_F else None),
            'fewest_hop_e2e'  : round(hp_e2e, 5),
            'routing_gain'    : round(e2e_F - hp_e2e, 5),
            'qkd_ready'       : e2e_F >= QKD_FIDELITY_THRESH,
            'entangled'       : e2e_F >= SWAP_FIDELITY_THRESH,
        }

        if apply_purification:
            if pair_i > 0 and pair_i % 20 == 0:
                print(f"    [routing] processed {pair_i}/{len(pairs)} pairs ...")
            pur = purify_path(G, path, n_repeats=purify_n_repeats)
            if pur is not None:
                purified_F = pur['purified_fidelity_mean']

                hop_Ts = [G[u][v]['transmittance']
                          for u, v in zip(path[:-1], path[1:])]
                eps_e2e_raw = _e2e_throughput_eps(
                    hop_Ts, n_swaps, cumulative_purification_yield=1.0)
                eps_e2e_purified = _e2e_throughput_eps(
                    hop_Ts, n_swaps,
                    cumulative_purification_yield=pur['mean_cumulative_yield'])

                hops_for_qber = list(zip(path[:-1], path[1:]))
                _d0 = G[hops_for_qber[0][0]][hops_for_qber[0][1]]['distance_km']
                _T0 = G[hops_for_qber[0][0]][hops_for_qber[0][1]]['transmittance']
                _rho_e2e = regenerate_hop_state(_d0, _T0)
                for (_u, _v) in hops_for_qber[1:]:
                    _di = G[_u][_v]['distance_km']
                    _Ti = G[_u][_v]['transmittance']
                    _rho_next = regenerate_hop_state(_di, _Ti)
                    _rho_e2e, _ = swap_bsm(_rho_e2e, _rho_next)
                _eZ_raw, _eX_raw = qber_from_rho(_rho_e2e)
                qber_e2e_raw = max(_eZ_raw, _eX_raw)

                _rho_pur_e2e, _p_pur = purify_bbpssw(
                    _rho_e2e, regenerate_hop_state(_d0, _T0)) \
                    if purified_F > PURIFICATION_F_LOW else (_rho_e2e, 1.0)
                if _rho_pur_e2e is not None and _p_pur > 0:
                    _rho_pur_e2e = _rho_pur_e2e
                else:
                    _rho_pur_e2e = _rho_e2e
                _eZ_pur, _eX_pur = qber_from_rho(_rho_pur_e2e)
                qber_e2e_purified = max(_eZ_pur, _eX_pur)

                skr_e2e_raw      = _secret_key_rate_bps(
                    qber_e2e_raw, eps_e2e_raw, e_phase=qber_e2e_raw)
                skr_e2e_purified = _secret_key_rate_bps(
                    qber_e2e_purified, eps_e2e_purified, e_phase=qber_e2e_purified)

                row.update({
                    'purified_e2e_fidelity'    : round(purified_F, 5),
                    'purified_e2e_fidelity_std': round(pur['purified_fidelity_std'], 5),
                    'purification_gain'        : round(purified_F - e2e_F, 5),
                    'purification_rounds_mean' : round(pur['mean_rounds_used'], 3),
                    'purification_yield_mean'  : round(pur['mean_cumulative_yield'], 4),
                    'purification_attempted_frac': round(
                        pur['fraction_purification_attempted'], 3),
                    'purified_qkd_ready'       : purified_F >= QKD_FIDELITY_THRESH,
                    'purified_entangled'       : purified_F >= SWAP_FIDELITY_THRESH,
                    'qber_e2e_raw'             : round(qber_e2e_raw, 5),
                    'qber_e2e_purified'        : round(qber_e2e_purified, 5),
                    'eps_e2e_raw'              : round(eps_e2e_raw, 3),
                    'eps_e2e_purified'         : round(eps_e2e_purified, 3),
                    'skr_e2e_raw_bps'          : round(skr_e2e_raw, 4),
                    'skr_e2e_purified_bps'     : round(skr_e2e_purified, 4),
                    'skr_gain_bps'             : round(skr_e2e_purified - skr_e2e_raw, 4),
                })

        rows.append(row)
        all_paths.append({**row,
                          'path_nodes': path,
                          'hop_fids'  : hop_fids})

    routing_table = pd.DataFrame(rows).sort_values('e2e_fidelity', ascending=False)
    return routing_table, all_paths


def run_and_visualise_routing(G, clusters):
    print("\n  Running Dijkstra entanglement-swapping routing engine ...")
    routing_table, all_paths = run_routing_engine(G)
    print(f"  Computed {len(routing_table)} optimal paths")

    if routing_table.empty:
        print("  [!] No routes found."); return routing_table

    print("\n" + "="*80)
    print("  DIJKSTRA ENTANGLEMENT-ROUTING TABLE  (top 20 by E2E fidelity)")
    print("="*80)
    cols_show = ['source','destination','hops','n_swaps','e2e_fidelity',
                 'total_dist_km','min_hop_fidelity','routing_gain',
                 'qkd_ready','entangled']
    if 'purified_e2e_fidelity' in routing_table.columns:
        cols_show += ['purified_e2e_fidelity', 'purification_gain',
                      'purified_qkd_ready', 'purified_entangled']
    print(routing_table[cols_show].head(20).to_string(index=False))

    n_qkd = routing_table['qkd_ready'].sum()
    n_ent = routing_table['entangled'].sum()
    n_tot = len(routing_table)
    print(f"\n  QKD-ready routes : {n_qkd}/{n_tot}  ({n_qkd/n_tot*100:.1f}%)")
    print(f"  Entangled routes : {n_ent}/{n_tot}  ({n_ent/n_tot*100:.1f}%)")
    if 'purified_qkd_ready' in routing_table.columns:
        n_qkd_p = routing_table['purified_qkd_ready'].sum()
        n_ent_p = routing_table['purified_entangled'].sum()
        mean_gain = routing_table['purification_gain'].mean()
        frac_attempted = routing_table['purification_attempted_frac'].mean()
        print(f"  -- after purification (DEJMPS/BBPSSW-twirled, adaptive) --")
        print(f"  QKD-ready routes (purified) : {n_qkd_p}/{n_tot}  ({n_qkd_p/n_tot*100:.1f}%)")
        print(f"  Entangled routes (purified) : {n_ent_p}/{n_tot}  ({n_ent_p/n_tot*100:.1f}%)")
        print(f"  Mean purification gain      : {mean_gain:+.5f}")
        print(f"  Mean fraction attempted     : {frac_attempted:.1%}")
    best = routing_table.iloc[0]
    print(f"  Best route       : {best['source']} -> {best['destination']}")
    print(f"    Path           : {best['path']}")
    print(f"    E2E Fidelity   : {best['e2e_fidelity']:.5f}")
    print(f"    Hops / Swaps   : {int(best['hops'])} / {int(best['n_swaps'])}")
    print(f"    Total distance : {best['total_dist_km']:.2f} km")
    print("="*80)

    routing_table.to_csv(f"{OUT}/routing_table_dijkstra.csv", index=False)
    print(f"  [✓] {OUT}/routing_table_dijkstra.csv")

    print("\n  Generating routing visualisations ...")
    fig_routing_heatmap(routing_table, G)
    fig_routing_gain(routing_table)
    fig_routing_all_paths(G, all_paths, clusters)
    if all_paths:
        fig_routing_hop_profiles(all_paths, top_n=min(8, len(all_paths)))
    if 'purified_e2e_fidelity' in routing_table.columns:
        fig_purification_comparison(routing_table)

    return routing_table
