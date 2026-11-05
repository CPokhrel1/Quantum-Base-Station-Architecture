import numpy as np

from .channel import jitter_vis
from .config import (
    ATM_SCALE_HEIGHT, BSM_HARDWARE_MS, BSM_LINEAR_OPTIC_CAP, CLASSICAL_PROC_MS,
    COHERENCE_ATM_DECAY, COHERENCE_BASE_MS, DETECTOR_EFFICIENCY, ETA_SWAP,
    FEEDFORWARD_MS, FSO_COUPLING_EFF, KATHMANDU_ALT, MAX_MOBILE_KM,
    PHOTON_EMISSION_PROB, SOURCE_REP_RATE_HZ, SPEED_OF_LIGHT, WAVELENGTH,
)

# Ref: Sangouard et al., Rev. Mod. Phys. 83, 33 (2011)
# Ref: Briegel et al., Phys. Rev. Lett. 81, 5932 (1998)
# Ref: Duan, Lukin, Cirac & Zoller, Nature 414, 413 (2001)


EXTENDED_COLS = [
    'prop_latency_ms',
    'herald_latency_ms',
    'bsm_latency_ms',
    'announce_latency_ms',
    'feedforward_latency_ms',
    'T_rtt_ms',
    'mean_attempts',
    'total_latency_ms',
    'attempts_per_second',
    'success_prob',
    'throughput_eps',
    'entanglement_gen_rate',
    'secret_key_rate_bps',
    'decoherence_time_ms',
    'memory_feasible',
    'link_budget_db',
    'doppler_shift_hz',
    'pointing_loss_db',
    'handover_prob',
    'effective_throughput_bps',
]


def _prop_latency_ms(dist_km: float) -> float:
    return (dist_km * 1e3 / SPEED_OF_LIGHT) * 1e3


def _herald_latency_ms(dist_km: float) -> float:
    return _prop_latency_ms(dist_km)


# Ref: Walmsley group, Nature 530, 313 (2016)
def _bsm_latency_ms() -> float:
    return BSM_HARDWARE_MS


def _announce_latency_ms() -> float:
    return CLASSICAL_PROC_MS


# Ref: Bradley et al., Phys. Rev. X 9, 031045 (2019)
def _feedforward_latency_ms() -> float:
    return FEEDFORWARD_MS


def _T_rtt_ms(dist_km: float) -> float:
    return (
        _prop_latency_ms(dist_km)
        + _herald_latency_ms(dist_km)
        + _bsm_latency_ms()
        + _announce_latency_ms()
    )


# Ref: Sangouard et al., Rev. Mod. Phys. 83, 33 (2011), Eq. 4.1
# Ref: Hong, Ou & Mandel, Phys. Rev. Lett. 59, 2044 (1987)
# Ref: Dikmelik & Davidson, Appl. Opt. 44, 4946 (2005)
# Ref: Vasylyev et al., Phys. Rev. Lett. 108, 220501 (2012)
def _success_prob(transmittance_val: float, hom_visibility: float = None) -> float:
    if hom_visibility is None:
        hom_visibility = jitter_vis()
    return (BSM_LINEAR_OPTIC_CAP
            * PHOTON_EMISSION_PROB
            * FSO_COUPLING_EFF
            * transmittance_val
            * (DETECTOR_EFFICIENCY ** 2)
            * hom_visibility)


def _mean_attempts(p_s: float) -> float:
    return 1.0 / max(p_s, 1e-15)


# Ref: Sangouard et al., Rev. Mod. Phys. 83, 33 (2011), Sec. IV
# Ref: Briegel et al., Phys. Rev. Lett. 81, 5932 (1998), Eq. 1
def _total_latency_ms(dist_km: float, transmittance_val: float) -> float:
    p_s   = _success_prob(transmittance_val)
    T_rtt = _T_rtt_ms(dist_km)
    return T_rtt * _mean_attempts(p_s) + _feedforward_latency_ms()


def _e2e_throughput_eps(hop_transmittances: list, n_swaps: int,
                         cumulative_purification_yield: float = 1.0,
                         eta_swap: float = ETA_SWAP) -> float:
    if not hop_transmittances:
        return 0.0
    per_hop_eps = [_throughput_eps(T) for T in hop_transmittances]
    bottleneck_eps = min(per_hop_eps)
    swap_efficiency_total = eta_swap ** max(n_swaps, 0)
    return bottleneck_eps * swap_efficiency_total * cumulative_purification_yield


def _throughput_eps(transmittance_val: float) -> float:
    p_s = _success_prob(transmittance_val)
    return SOURCE_REP_RATE_HZ * p_s


# Ref: Shor & Preskill, Phys. Rev. Lett. 85, 441 (2000)
# Ref: Ekert, Phys. Rev. Lett. 67, 661 (1991)
# Ref: Gisin et al., Rev. Mod. Phys. 74, 145 (2002), Sec. IV.A
def _secret_key_rate_bps(qber_val: float, eps: float,
                          e_phase: float = None) -> float:
    def H2(p):
        p = max(0.0, min(1.0, p))
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

    e_bit   = max(0.0, qber_val)
    e_ph    = max(0.0, e_phase if e_phase is not None else qber_val)
    key_frac = 1.0 - H2(e_bit) - H2(e_ph)
    if key_frac <= 0:
        return 0.0
    return max(0.0, eps * 0.5 * key_frac)


# Ref: Herbschleb et al., Nat. Commun. 10, 3766 (2019)
# Ref: Degen et al., Rev. Mod. Phys. 89, 035002 (2017)
def _decoherence_time_ms(dist_km: float) -> float:
    eta_atm = np.exp(-KATHMANDU_ALT / ATM_SCALE_HEIGHT)
    return max(COHERENCE_BASE_MS / (1.0 + COHERENCE_ATM_DECAY * eta_atm * dist_km),
               0.01)


def _link_budget_db(transmittance_val: float) -> float:
    eff = max(transmittance_val * DETECTOR_EFFICIENCY ** 2, 1e-15)
    return -10.0 * np.log10(eff)


def _doppler_shift(speed_kmh: float) -> float:
    return (speed_kmh / 3.6) / WAVELENGTH


# Ref: Andrews & Phillips, Laser Beam Propagation through Random Media (2005)
# Ref: Farid & Hranilovic, J. Lightw. Technol. 25, 1702 (2007)
def _pointing_loss_db(speed_kmh: float, dist_km: float) -> float:
    if speed_kmh < 0.01:
        return 0.0
    v_ms      = speed_kmh / 3.6
    theta_div = WAVELENGTH / 0.10
    sigma_theta = v_ms / max(dist_km * 1e3, 1.0)
    return min(8.686 * (sigma_theta / theta_div) ** 2, 45.0)


# Ref: 3GPP TR 38.821
def _handover_prob(speed_kmh: float, T_rtt_ms: float) -> float:
    R_cell_m  = MAX_MOBILE_KM * 1e3
    v_ms      = speed_kmh / 3.6
    lambda_ho = v_ms / R_cell_m
    T_s       = T_rtt_ms * 1e-3
    return 1.0 - np.exp(-lambda_ho * T_s)


def add_extended_metrics(df, is_mobile=False):
    df = df.copy()
    d  = df['distance_km']
    T  = df['transmittance']

    df['prop_latency_ms']        = d.apply(_prop_latency_ms)
    df['herald_latency_ms']      = d.apply(_herald_latency_ms)
    df['bsm_latency_ms']         = BSM_HARDWARE_MS
    df['announce_latency_ms']    = _announce_latency_ms()
    df['feedforward_latency_ms'] = FEEDFORWARD_MS
    df['T_rtt_ms']               = d.apply(_T_rtt_ms)

    df['success_prob']           = T.apply(_success_prob)
    df['mean_attempts']          = df['success_prob'].apply(_mean_attempts)

    df['total_latency_ms'] = df.apply(
        lambda r: _total_latency_ms(r['distance_km'], r['transmittance']), axis=1)

    df['attempts_per_second']    = SOURCE_REP_RATE_HZ
    df['throughput_eps']         = T.apply(_throughput_eps)
    df['entanglement_gen_rate']  = df['throughput_eps']

    df['secret_key_rate_bps'] = df.apply(
        lambda r: _secret_key_rate_bps(r['qber'], r['throughput_eps'],
                                        e_phase=r['qber']), axis=1)

    df['decoherence_time_ms'] = d.apply(_decoherence_time_ms)
    df['memory_feasible'] = (
        df['decoherence_time_ms'] >= df['T_rtt_ms'] * 10
    ).astype(int)

    df['link_budget_db'] = T.apply(_link_budget_db)

    if is_mobile and 'speed_kmh' in df.columns:
        df['doppler_shift_hz']  = df['speed_kmh'].apply(_doppler_shift)
        df['pointing_loss_db']  = df.apply(
            lambda r: _pointing_loss_db(r['speed_kmh'], r['distance_km']), axis=1)
        df['handover_prob']     = df.apply(
            lambda r: _handover_prob(r['speed_kmh'], r['T_rtt_ms']), axis=1)
        df['effective_throughput_bps'] = (df['secret_key_rate_bps']
                                          * (1.0 - df['handover_prob']))
    else:
        df['doppler_shift_hz']         = 0.0
        df['pointing_loss_db']         = 0.0
        df['handover_prob']            = 0.0
        df['effective_throughput_bps'] = df['secret_key_rate_bps']

    return df
