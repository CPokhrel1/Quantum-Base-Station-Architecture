from itertools import combinations

import networkx as nx
import pandas as pd

from .channel import haversine_km
from .config import MAX_LINK_KM, MAX_MOBILE_KM
from .generation import simulate_link
from .link_metrics import _e2e_throughput_eps, _secret_key_rate_bps
from .purification import purify_link
from .quantum_metrics import qber_from_rho
from .swapping import regenerate_hop_state


def build_bts_graph(clusters):
    G = nx.Graph()
    for _,r in clusters.iterrows():
        G.add_node(r['name'], pos=(r['lon'],r['lat']),
                   n_cells=r['n_cells'], ntype=r['type'],
                   samples=r['total_samples'],
                   gsm=r['gsm'], umts=r['umts'], lte=r['lte'],
                   node_class='bts')
    rows = [r for _,r in clusters.iterrows()]
    pairs = list(combinations(rows,2))
    print(f"  BTS pairs to simulate: {len(pairs)}")
    for a,b in pairs:
        d = haversine_km(a['lat'],a['lon'],b['lat'],b['lon'])
        if d <= MAX_LINK_KM:
            res = simulate_link(a['lat'],a['lon'],b['lat'],b['lon'])
            G.add_edge(a['name'],b['name'], **res, link_type='bts-bts')
    return G


def assign_mobile_to_bts(mobile_df, clusters):
    assignments = []
    for _,mob in mobile_df.iterrows():
        dists = []
        for _,bts in clusters.iterrows():
            d = haversine_km(mob['lat'],mob['lon'],bts['lat'],bts['lon'])
            if d <= MAX_MOBILE_KM:
                dists.append((d, bts['name'], bts['lat'], bts['lon']))
        dists.sort(key=lambda x: x[0])
        assignments.append(dists[:2])
    return assignments


def simulate_mobile_links(mobile_df, assignments, clusters,
                           apply_purification: bool = True,
                           purify_n_repeats: int = 10):
    bts_lookup = {r['name']: r for _,r in clusters.iterrows()}
    records = []
    print(f"  Simulating mobile node links ...")
    if apply_purification:
        n_links_est = sum(len(a) for a in assignments)
        print(f"    purification enabled for ~{n_links_est} mobile links "
              f"(purify_n_repeats={purify_n_repeats}) -- may take 1-2 minutes.")

    for idx, (mob_idx, mob) in enumerate(mobile_df.iterrows()):
        for (d_km, bts_name, bts_lat, bts_lon) in assignments[idx]:
            res = simulate_link(mob['lat'], mob['lon'],
                                bts_lat, bts_lon,
                                speed_kmh=mob['speed_kmh'],
                                is_mobile=True)

            if apply_purification:
                T_eff = res['transmittance']
                pur = purify_link(d_km, T_eff, n_repeats=purify_n_repeats)
                purified_F = pur['purified_fidelity_mean']
                raw_F = res['fidelity']

                eps_raw = _e2e_throughput_eps([T_eff], n_swaps=0,
                                              cumulative_purification_yield=1.0)
                eps_purified = _e2e_throughput_eps([T_eff], n_swaps=0,
                                                   cumulative_purification_yield=pur['mean_cumulative_yield'])
                _rho_pur_sample = regenerate_hop_state(d_km, T_eff)
                _eZ_pur, _eX_pur = qber_from_rho(_rho_pur_sample)
                qber_purified = max(_eZ_pur, _eX_pur)
                skr_raw = _secret_key_rate_bps(res['qber'],
                                               eps_raw,
                                               e_phase=res['qber'])
                skr_purified = _secret_key_rate_bps(qber_purified,
                                                     eps_purified,
                                                     e_phase=qber_purified)

                res.update({
                    'purified_fidelity'    : round(purified_F, 5),
                    'purified_fidelity_std': round(pur['purified_fidelity_std'], 5),
                    'purification_gain'    : round(purified_F - raw_F, 5),
                    'purification_rounds_mean': round(pur['mean_rounds_used'], 3),
                    'purification_yield_mean' : round(pur['mean_cumulative_yield'], 4),
                    'purification_attempted_frac': round(
                        pur['fraction_purification_attempted'], 3),
                    'purified_qber'        : round(qber_purified, 5),
                    'eps_raw'              : round(eps_raw, 3),
                    'eps_purified'         : round(eps_purified, 3),
                    'skr_raw_bps'          : round(skr_raw, 4),
                    'skr_purified_bps'     : round(skr_purified, 4),
                    'skr_gain_bps'         : round(skr_purified - skr_raw, 4),
                })

            records.append({
                'mobile_id'  : mob['mobile_id'],
                'mob_type'   : mob['mob_type'],
                'speed_kmh'  : mob['speed_kmh'],
                'heading_deg': mob['heading_deg'],
                'bts_name'   : bts_name,
                'mob_lat'    : mob['lat'],
                'mob_lon'    : mob['lon'],
                **res,
                'link_type'  : 'mobile-bts',
            })
        if (idx+1) % 20 == 0:
            print(f"    ... {idx+1}/{len(mobile_df)} mobile nodes done")
    return pd.DataFrame(records)
