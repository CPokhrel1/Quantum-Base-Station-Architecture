from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from qutip import ket2dm, qeye, Qobj, sigmax, sigmay, sigmaz, tensor

from .config import PURIFICATION_F_HIGH, PURIFICATION_F_LOW, PURIFICATION_MAX_ROUNDS
from .link_metrics import _e2e_throughput_eps, _secret_key_rate_bps
from .quantum_metrics import metrics, qber_from_rho
from .swapping import _BELL_KETS, regenerate_hop_state, swap_bsm


_CNOT_2Q = Qobj(np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]]), dims=[[2, 2], [2, 2]])
_PROJ0 = ket2dm(Qobj([[1], [0]]))
_PROJ1 = ket2dm(Qobj([[0], [1]]))
_H_GATE = Qobj(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
_TWIRL_PAULIS = [qeye(2), sigmax(), sigmay(), sigmaz()]
_BELL_BASIS_MATRIX = Qobj(
    np.column_stack([bk.full().flatten() for bk in _BELL_KETS]),
    dims=[[2, 2], [2, 2]])


def bilateral_pauli_twirl_to_werner(rho: Qobj) -> Qobj:
    acc = None
    for P in _TWIRL_PAULIS:
        U = tensor(P, P)
        term = U * rho * U.dag()
        acc = term if acc is None else acc + term
    rho_bd = acc / 4.0

    rho_bell_diag = _BELL_BASIS_MATRIX.dag() * rho_bd * _BELL_BASIS_MATRIX
    diag = np.real(rho_bell_diag.full().diagonal())
    F = diag[0]
    rest_avg = diag[1:].mean()
    new_diag = np.array([F, rest_avg, rest_avg, rest_avg])
    rho_bell_diag_sym = Qobj(np.diag(new_diag), dims=[[2, 2], [2, 2]])
    return _BELL_BASIS_MATRIX * rho_bell_diag_sym * _BELL_BASIS_MATRIX.dag()


# Ref: Bennett et al., Phys. Rev. A 53, 2046 (1996)
def purify_bbpssw(rho1: Qobj, rho2: Qobj):
    full = tensor(rho1, rho2)
    full_p = full.permute([0, 2, 1, 3])
    U = tensor(_CNOT_2Q, _CNOT_2Q)
    full_p = U * full_p * U.dag()

    I = qeye(2)
    keep_projectors = [tensor(I, _PROJ0, I, _PROJ0), tensor(I, _PROJ1, I, _PROJ1)]
    p_success = 0.0
    acc = None
    for P in keep_projectors:
        unnorm = P * full_p * P
        p_success += unnorm.tr().real
        acc = unnorm if acc is None else acc + unnorm

    if p_success < 1e-12:
        return None, 0.0
    rho_out = acc.ptrace([0, 2]) / p_success
    return rho_out, p_success


def purify_bbpssw_twirled(rho1: Qobj, rho2: Qobj):
    rho1_w = bilateral_pauli_twirl_to_werner(rho1)
    rho2_w = bilateral_pauli_twirl_to_werner(rho2)
    return purify_bbpssw(rho1_w, rho2_w)


# Ref: Deutsch et al., Phys. Rev. Lett. 77, 2818 (1996)
# Ref: Jansen et al., arXiv:2509.03667
def dejmps_purify(rho1: Qobj, rho2: Qobj):
    Urot = tensor(_H_GATE, _H_GATE)
    rho1_r = Urot * rho1 * Urot.dag()
    rho2_r = Urot * rho2 * Urot.dag()

    full = tensor(rho1_r, rho2_r)
    full_p = full.permute([0, 2, 1, 3])
    U = tensor(_CNOT_2Q, _CNOT_2Q)
    full_p = U * full_p * U.dag()

    I = qeye(2)
    keep_projectors = [tensor(I, _PROJ0, I, _PROJ0), tensor(I, _PROJ1, I, _PROJ1)]
    p_success = 0.0
    acc = None
    for P in keep_projectors:
        unnorm = P * full_p * P
        p_success += unnorm.tr().real
        acc = unnorm if acc is None else acc + unnorm

    if p_success < 1e-12:
        return None, 0.0
    rho_out_rotated_frame = acc.ptrace([0, 2]) / p_success
    rho_out = Urot.dag() * rho_out_rotated_frame * Urot
    return rho_out, p_success


def purify_precheck_gain(rho1: Qobj, rho2: Qobj, protocol: str = 'dejmps',
                          n_samples: int = 8):
    protocol_fn = dejmps_purify if protocol == 'dejmps' else purify_bbpssw
    F1, _, _, _ = metrics(rho1)
    rho_out, p_success = protocol_fn(rho1, rho2)
    if rho_out is None:
        return None
    F_out, _, _, _ = metrics(rho_out)
    return F_out - F1


def purify_adaptive(state_sampler, protocol: str = 'auto',
                     max_rounds: int = PURIFICATION_MAX_ROUNDS,
                     min_gain: float = 0.005, n_precheck: int = 8,
                     f_floor: float = PURIFICATION_F_LOW):
    rho = state_sampler()
    F_raw, _, _, _ = metrics(rho)

    result = {
        'raw_fidelity': F_raw, 'purified_fidelity': F_raw,
        'protocol_used': None, 'n_rounds_used': 0,
        'cumulative_yield': 1.0, 'pairs_consumed': 1,
        'was_purified': False,
        'precheck_gain_dejmps': None, 'precheck_gain_bbpssw_twirled': None,
    }

    if F_raw < f_floor:
        return result

    if protocol == 'auto':
        gains_dejmps = [purify_precheck_gain(state_sampler(), state_sampler(),
                                              protocol='dejmps')
                        for _ in range(n_precheck)]
        gains_dejmps = [g for g in gains_dejmps if g is not None]
        mean_gain_dejmps = float(np.mean(gains_dejmps)) if gains_dejmps else -1.0

        gains_bbtw = []
        for _ in range(n_precheck):
            r1, r2 = state_sampler(), state_sampler()
            r1w, r2w = bilateral_pauli_twirl_to_werner(r1), bilateral_pauli_twirl_to_werner(r2)
            g = purify_precheck_gain(r1w, r2w, protocol='bbpssw')
            if g is not None:
                gains_bbtw.append(g)
        mean_gain_bbtw = float(np.mean(gains_bbtw)) if gains_bbtw else -1.0

        result['precheck_gain_dejmps'] = mean_gain_dejmps
        result['precheck_gain_bbpssw_twirled'] = mean_gain_bbtw

        if max(mean_gain_dejmps, mean_gain_bbtw) <= 0:
            return result

        chosen = 'dejmps' if mean_gain_dejmps >= mean_gain_bbtw else 'bbpssw_twirled'
    else:
        chosen = protocol

    result['protocol_used'] = chosen
    protocol_fn = (dejmps_purify if chosen == 'dejmps'
                   else lambda a, b: purify_bbpssw(
                       bilateral_pauli_twirl_to_werner(a),
                       bilateral_pauli_twirl_to_werner(b)))

    F_current = F_raw
    cumulative_yield = 1.0
    rounds_used = 0
    for _r in range(max_rounds):
        rho_fresh = state_sampler()
        rho_out, p_success = protocol_fn(rho, rho_fresh)
        if rho_out is None or p_success < 1e-9:
            break
        F_new, _, _, _ = metrics(rho_out)
        if F_new - F_current < min_gain:
            break
        rho, F_current = rho_out, F_new
        cumulative_yield *= p_success
        rounds_used += 1
        if F_current > PURIFICATION_F_HIGH:
            break

    result.update({
        'purified_fidelity': F_current,
        'n_rounds_used': rounds_used,
        'cumulative_yield': cumulative_yield if rounds_used > 0 else 1.0,
        'pairs_consumed': 2 ** rounds_used if rounds_used > 0 else 1,
        'was_purified': rounds_used > 0,
    })
    return result


def purify_chain(d_km: float, T: float, **kwargs):
    return purify_adaptive(lambda: regenerate_hop_state(d_km, T), **kwargs)


# Ref: Briegel et al., Phys. Rev. Lett. 81, 5932 (1998)
def purify_path(G: nx.Graph, path: list, n_repeats: int = 20, **purify_kwargs):
    hops = list(zip(path[:-1], path[1:]))
    if not hops:
        return None

    def one_e2e_realization():
        d0, T0 = G[hops[0][0]][hops[0][1]]['distance_km'], G[hops[0][0]][hops[0][1]]['transmittance']
        rho_acc = regenerate_hop_state(d0, T0)
        for (u, v) in hops[1:]:
            d_i, T_i = G[u][v]['distance_km'], G[u][v]['transmittance']
            rho_next = regenerate_hop_state(d_i, T_i)
            rho_acc, _ = swap_bsm(rho_acc, rho_next)
        return rho_acc

    return _aggregate_purification_repeats(one_e2e_realization, n_repeats, **purify_kwargs)


def purify_link(d_km: float, T: float, n_repeats: int = 12, **purify_kwargs):
    return _aggregate_purification_repeats(
        lambda: regenerate_hop_state(d_km, T), n_repeats, **purify_kwargs)


def _aggregate_purification_repeats(state_sampler, n_repeats: int, **purify_kwargs):
    raw_Fs, pur_Fs, rounds_list, yields_list, protocols_used = [], [], [], [], []
    n_attempted = 0

    for _ in range(n_repeats):
        res = purify_adaptive(state_sampler, **purify_kwargs)
        raw_Fs.append(res['raw_fidelity'])
        pur_Fs.append(res['purified_fidelity'])
        rounds_list.append(res['n_rounds_used'])
        yields_list.append(res['cumulative_yield'])
        if res['was_purified']:
            n_attempted += 1
            protocols_used.append(res['protocol_used'])

    protocol_counts = dict(Counter(protocols_used))

    return {
        'raw_fidelity_mean': float(np.mean(raw_Fs)),
        'raw_fidelity_std': float(np.std(raw_Fs)),
        'purified_fidelity_mean': float(np.mean(pur_Fs)),
        'purified_fidelity_std': float(np.std(pur_Fs)),
        'mean_rounds_used': float(np.mean(rounds_list)),
        'mean_cumulative_yield': float(np.mean(yields_list)),
        'fraction_purification_attempted': n_attempted / n_repeats if n_repeats > 0 else 0.0,
        'protocol_usage_counts': protocol_counts,
        'n_repeats': n_repeats,
    }


def add_single_edge_purification(df: pd.DataFrame, purify_n_repeats: int = 10) -> pd.DataFrame:
    df = df.copy()
    purified_F_list, purified_F_std_list = [], []
    rounds_list, yield_list, attempted_list = [], [], []
    qber_pur_list, eps_raw_list, eps_pur_list = [], [], []
    skr_raw_list, skr_pur_list = [], []

    for _, row in df.iterrows():
        d_km = row['distance_km']
        T_eff = row['transmittance']
        pur = purify_link(d_km, T_eff, n_repeats=purify_n_repeats)
        purified_F = pur['purified_fidelity_mean']

        eps_raw = _e2e_throughput_eps([T_eff], n_swaps=0,
                                      cumulative_purification_yield=1.0)
        eps_pur = _e2e_throughput_eps([T_eff], n_swaps=0,
                                      cumulative_purification_yield=pur['mean_cumulative_yield'])
        _rho_sample = regenerate_hop_state(d_km, T_eff)
        _eZ, _eX = qber_from_rho(_rho_sample)
        qber_pur = max(_eZ, _eX)
        q_raw = row['qber']
        skr_raw = _secret_key_rate_bps(q_raw, eps_raw, e_phase=q_raw)
        skr_pur = _secret_key_rate_bps(qber_pur, eps_pur, e_phase=qber_pur)

        purified_F_list.append(round(purified_F, 5))
        purified_F_std_list.append(round(pur['purified_fidelity_std'], 5))
        rounds_list.append(round(pur['mean_rounds_used'], 3))
        yield_list.append(round(pur['mean_cumulative_yield'], 4))
        attempted_list.append(round(pur['fraction_purification_attempted'], 3))
        qber_pur_list.append(round(qber_pur, 5))
        eps_raw_list.append(round(eps_raw, 3))
        eps_pur_list.append(round(eps_pur, 3))
        skr_raw_list.append(round(skr_raw, 4))
        skr_pur_list.append(round(skr_pur, 4))

    df['purified_fidelity']     = purified_F_list
    df['purified_fidelity_std'] = purified_F_std_list
    df['purification_gain']     = df['purified_fidelity'] - df['fidelity']
    df['purification_rounds_mean']    = rounds_list
    df['purification_yield_mean']     = yield_list
    df['purification_attempted_frac'] = attempted_list
    df['purified_qber']  = qber_pur_list
    df['eps_raw']         = eps_raw_list
    df['eps_purified']    = eps_pur_list
    df['skr_raw_bps']     = skr_raw_list
    df['skr_purified_bps'] = skr_pur_list
    df['skr_gain_bps']    = df['skr_purified_bps'] - df['skr_raw_bps']
    return df
