"""
Quantum BTS Architecture -- Kathmandu Entanglement Distribution
Extended with Mobile Nodes (50-200, range <= 4500 m)
All stats + all visualizations (separate PNGs)
"""

import pandas as pd

from quantum_bts.config import MAX_MOBILE_KM, N_MOBILE
from quantum_bts.data import add_mobile_nodes, load_data
from quantum_bts.figures.dashboards import (
    fig_dashboard_bts, fig_dashboard_comparison, fig_dashboard_mobile,
)
from quantum_bts.figures.e2e import fig_e2e
from quantum_bts.figures.extended import (
    fig_extended_metrics_dashboard, fig_latency_overview, fig_link_budget,
    fig_purified_skr_comparison, fig_skr_analysis,
)
from quantum_bts.figures.mobile import (
    fig_fidelity_vs_dist_speed, fig_mobile_fidelity_dist, fig_mobile_type_comparison,
    fig_qber_vs_speed,
)
from quantum_bts.figures.network import (
    fig_bts_load_count, fig_bts_load_metrics, fig_coverage_map, fig_fidelity_heatmap,
    fig_mobile_spatial, fig_network_map, fig_transmittance,
)
from quantum_bts.figures.noise import fig_noise_breakdown
from quantum_bts.link_metrics import add_extended_metrics, EXTENDED_COLS
from quantum_bts.network import (
    assign_mobile_to_bts, build_bts_graph, simulate_mobile_links,
)
from quantum_bts.purification import add_single_edge_purification
from quantum_bts.routing import run_and_visualise_routing
from quantum_bts.stats import compute_all_stats


def main():
    print("="*70)
    print("  Quantum BTS + Mobile Node Simulation -- Kathmandu Valley")
    print("  With Dijkstra Entanglement-Swapping Routing + Extended Metrics")
    print("="*70)

    df_raw, clusters = load_data()
    print(f"  Towers: {len(df_raw)} | BTS sites: {len(clusters)}")

    print(f"\n  Generating {N_MOBILE} mobile nodes (range <= {MAX_MOBILE_KM} km) ...")
    mobile_df = add_mobile_nodes(clusters, N_MOBILE)
    for t, c in mobile_df['mob_type'].value_counts().items():
        print(f"    {t}: {c}")

    print("\n  Building BTS graph ...")
    G = build_bts_graph(clusters)
    print(f"  BTS Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n  Assigning mobile nodes to BTS ...")
    assignments = assign_mobile_to_bts(mobile_df, clusters)
    connected = sum(1 for a in assignments if a)
    print(f"  Mobile nodes with >=1 BTS in range: {connected}/{N_MOBILE}")

    print("\n  Simulating mobile quantum links ...")
    mob_links = simulate_mobile_links(mobile_df, assignments, clusters)
    print(f"  Total mobile links simulated: {len(mob_links)}")

    print("\n  Computing extended metrics (latency, SKR, decoherence time …) …")
    mob_links  = add_extended_metrics(mob_links,  is_mobile=True)
    edges      = list(G.edges(data=True))
    if edges:
        bts_link_df = pd.DataFrame([
            {**{'source':u,'destination':v,'link_type':'bts-bts'},
             **{k:round(v2,5) if isinstance(v2,float) else v2 for k,v2 in d.items()}}
            for u,v,d in edges])
        print(f"    Purifying {len(bts_link_df)} individual BTS-BTS edges "
              f"(single-hop, symmetric with mobile links) ...")
        bts_link_df = add_single_edge_purification(bts_link_df, purify_n_repeats=10)
        bts_link_df = add_extended_metrics(bts_link_df, is_mobile=False)
        for _, row in bts_link_df.iterrows():
            u, v = row['source'], row['destination']
            if G.has_edge(u, v):
                for col in EXTENDED_COLS:
                    G[u][v][col] = row[col]
    else:
        bts_link_df = pd.DataFrame()

    print("\n  Generating all figures …")
    fig_network_map(G, df_raw, clusters, mobile_df, assignments, mob_links)
    fig_mobile_spatial(mobile_df, clusters)
    fig_mobile_fidelity_dist(mob_links)
    fig_fidelity_vs_dist_speed(mob_links)
    fig_mobile_type_comparison(mob_links)
    fig_coverage_map(mob_links, mobile_df, clusters)
    fig_bts_load_count(mob_links)
    fig_bts_load_metrics(mob_links)
    fig_noise_breakdown()
    fig_qber_vs_speed(mob_links)
    fig_fidelity_heatmap(G, clusters)
    fig_transmittance(mob_links, G)
    fig_e2e(G, clusters)
    fig_dashboard_bts(G)
    fig_dashboard_mobile(mob_links)
    fig_dashboard_comparison(G, mob_links)

    fig_latency_overview(mob_links, bts_link_df)
    fig_skr_analysis(mob_links, bts_link_df)
    fig_purified_skr_comparison(mob_links, bts_link_df)
    fig_link_budget(mob_links, bts_link_df)
    fig_extended_metrics_dashboard(mob_links, bts_link_df)

    routing_table = run_and_visualise_routing(G, clusters)

    print("\n  Computing all statistics …")
    compute_all_stats(G, mob_links, mobile_df, routing_table=routing_table)

    print("\n  All outputs saved to /mnt/user-data/outputs/")
    print("="*70)


if __name__ == "__main__":
    main()
