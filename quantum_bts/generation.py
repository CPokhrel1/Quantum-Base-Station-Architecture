import numpy as np
from qutip import bell_state, ket2dm, Qobj

from .channel import (
    AtmModel, jitter_vis, link_dist_row, loss_ch, mobile_extra_loss, phase_noise,
    pol_drift, transmittance,
)
from .config import (
    DARK_COUNT_RATE, DARK_COUNT_WINDOW_NS, HERALD_ABS_CAP, HERALD_CAP_MULTIPLE, N_MC,
    RNG, SPEED_OF_LIGHT,
)
from .link_metrics import _success_prob
from .quantum_metrics import metrics


def dark_count_prob(window_ns: float = DARK_COUNT_WINDOW_NS,
                     rate_hz: float = DARK_COUNT_RATE) -> float:
    window_s = window_ns * 1e-9
    p_one = 1 - np.exp(-rate_hz * window_s)
    return 1 - (1 - p_one) ** 2


# Ref: Barrett & Kok, Phys. Rev. A 71, 060310 (2005)
def heralded_bsm_trial(d_km, T, atm_model, dark_p=None, rng=None):
    rng = rng if rng is not None else RNG
    if dark_p is None:
        dark_p = dark_count_prob()

    transit = d_km * 1e3 / SPEED_OF_LIGHT
    phi0 = ket2dm(bell_state("00"))
    rho = atm_model.decohere(phi0, transit)
    rho = pol_drift(rho, d_km)
    rho = phase_noise(rho, d_km)
    rho = loss_ch(rho, T)

    if rng.random() < dark_p:
        mixed = Qobj(np.eye(4) / 4.0, dims=[[2, 2], [2, 2]])
        return mixed, False

    return rho, True


def simulate_heralded_generation(d_km, T, speed_kmh=0.0, is_mobile=False,
                                  n_trials=N_MC, cap_multiple=HERALD_CAP_MULTIPLE):
    T_eff = T * mobile_extra_loss(speed_kmh, d_km) if is_mobile else T
    atm = AtmModel()
    dark_p = dark_count_prob()
    V = jitter_vis()
    p_s = max(_success_prob(T_eff), 1e-12)
    max_attempts = min(int(cap_multiple / p_s), HERALD_ABS_CAP)

    Fs, Cs, Qs, Ns = [], [], [], []
    attempts_list = []
    n_dark_events = 0
    n_capped = 0

    for _ in range(n_trials):
        attempts = RNG.geometric(p_s)
        if attempts > max_attempts:
            n_capped += 1
            continue

        rho_final, is_genuine = heralded_bsm_trial(d_km, T_eff, atm, dark_p=dark_p, rng=RNG)

        attempts_list.append(attempts)
        tr = rho_final.tr().real
        if tr > 0:
            rho_final = rho_final / tr
        F, C, Q, N = metrics(rho_final)
        if not is_genuine:
            n_dark_events += 1
        Fs.append(F); Cs.append(C); Qs.append(Q); Ns.append(N)

    n_delivered = len(Fs)
    yield_frac = n_delivered / n_trials if n_trials > 0 else 0.0

    if n_delivered == 0:
        return {
            'distance_km': d_km, 'transmittance': T_eff,
            'fidelity': 0.0, 'fidelity_std': 0.0,
            'concurrence': 0.0, 'qber': 0.5, 'negativity': 0.0,
            'jitter_vis': V,
            'success_prob_empirical': 0.0,
            'mean_attempts_empirical': float(max_attempts),
            'yield_fraction': 0.0,
            'n_dark_count_events': n_dark_events,
            'n_capped_trials': n_capped,
        }

    return {
        'distance_km': d_km, 'transmittance': T_eff,
        'fidelity': float(np.mean(Fs)), 'fidelity_std': float(np.std(Fs)),
        'concurrence': float(np.mean(Cs)), 'qber': float(np.mean(Qs)),
        'negativity': float(np.mean(Ns)),
        'jitter_vis': V,
        'success_prob_empirical': float(1.0 / np.mean(attempts_list)) if attempts_list else 0.0,
        'mean_attempts_empirical': float(np.mean(attempts_list)) if attempts_list else float(max_attempts),
        'yield_fraction': yield_frac,
        'n_dark_count_events': n_dark_events,
        'n_capped_trials': n_capped,
    }


def simulate_link(a_lat, a_lon, b_lat, b_lon,
                  speed_kmh=0.0, is_mobile=False):
    d = link_dist_row(a_lat, a_lon, b_lat, b_lon)
    T = transmittance(d)
    return simulate_heralded_generation(d, T, speed_kmh=speed_kmh,
                                         is_mobile=is_mobile, n_trials=N_MC)
