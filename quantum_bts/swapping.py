import networkx as nx
import numpy as np
from qutip import bell_state, ket2dm, qeye, Qobj, sigmax, sigmay, sigmaz, tensor

from .channel import AtmModel
from .config import ETA_SWAP, RNG
from .generation import dark_count_prob, heralded_bsm_trial
from .quantum_metrics import metrics


def swap_fidelity(F1: float, F2: float) -> float:
    return F1 * F2 + (1 - F1) * (1 - F2) / 9.0


def e2e_fidelity(path_fids: list) -> float:
    F = path_fids[0]
    for Fi in path_fids[1:]:
        F = swap_fidelity(F, Fi)
    return F


_BELL_KETS        = [bell_state(s) for s in ('00', '01', '10', '11')]
_BELL_PROJECTORS  = [tensor(qeye(2), ket2dm(bk), qeye(2)) for bk in _BELL_KETS]
_BELL_CORRECTIONS = {0: qeye(2), 1: sigmaz(), 2: sigmax(), 3: 1j * sigmay()}


def swap_bsm(rho_AB: Qobj, rho_BC: Qobj, eta_swap: float = ETA_SWAP):
    full = tensor(rho_AB, rho_BC)
    probs = []
    for P in _BELL_PROJECTORS:
        unnorm = P * full * P
        probs.append(max(unnorm.tr().real, 0.0))
    total = sum(probs)
    if total < 1e-14:
        return Qobj(np.eye(4) / 4.0, dims=[[2, 2], [2, 2]]), -1

    probs = [p / total for p in probs]
    outcome = int(RNG.choice(4, p=probs))

    P = _BELL_PROJECTORS[outcome]
    unnorm = P * full * P
    rho_AC = unnorm.ptrace([0, 3]) / unnorm.tr().real

    U = tensor(qeye(2), _BELL_CORRECTIONS[outcome])
    rho_AC = U * rho_AC * U.dag()

    if eta_swap < 1.0:
        mixed = Qobj(np.eye(4) / 4.0, dims=[[2, 2], [2, 2]])
        rho_AC = eta_swap * rho_AC + (1.0 - eta_swap) * mixed

    return rho_AC, outcome


def regenerate_hop_state(d_km: float, T: float) -> Qobj:
    atm = AtmModel()
    dark_p = dark_count_prob()
    rho, _is_genuine = heralded_bsm_trial(d_km, T, atm, dark_p=dark_p, rng=RNG)
    tr = rho.tr().real
    return rho / tr if tr > 0 else rho


def real_e2e_fidelity_for_path(G: nx.Graph, path: list, n_repeats: int = 20):
    hops = list(zip(path[:-1], path[1:]))
    if not hops:
        return 0.0, 0.0, []

    Fs = []
    for _ in range(n_repeats):
        d0, T0 = G[hops[0][0]][hops[0][1]]['distance_km'], G[hops[0][0]][hops[0][1]]['transmittance']
        rho_acc = regenerate_hop_state(d0, T0)
        for (u, v) in hops[1:]:
            d_i, T_i = G[u][v]['distance_km'], G[u][v]['transmittance']
            rho_next = regenerate_hop_state(d_i, T_i)
            rho_acc, _outcome = swap_bsm(rho_acc, rho_next)
        F, _, _, _ = metrics(rho_acc)
        Fs.append(F)

    return float(np.mean(Fs)), float(np.std(Fs)), Fs
