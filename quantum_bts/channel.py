import numpy as np
from scipy.linalg import expm
from scipy.special import erf
from qutip import Qobj

from .config import (
    ATM_LOSS_SEA, ATM_SCALE_HEIGHT, COINC_WINDOW_NS, KATHMANDU_ALT, LASER_LINEWIDTH,
    POLARIZ_DRIFT_SD, RNG, SPEED_OF_LIGHT, TIMING_JITTER_PS, WAVELENGTH,
)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1,p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1); dl = np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def link_dist_row(a_lat,a_lon,b_lat,b_lon):
    return haversine_km(a_lat,a_lon,b_lat,b_lon)


class AtmModel:
    def __init__(self, alt=KATHMANDU_ALT):
        d = np.exp(-alt/ATM_SCALE_HEIGHT)
        self.g1 = d/0.8e-3; self.g2 = d/0.4e-3

    def decohere(self, rho, t):
        p1 = 1-np.exp(-t*self.g1)
        K0 = Qobj([[1,0],[0,np.sqrt(1-p1)]])
        K1 = Qobj([[0,np.sqrt(p1)],[0,0]])
        p2 = 0.5*(1-np.exp(-t*self.g2))
        D0 = Qobj([[1,0],[0,np.sqrt(1-2*p2)]])
        D1 = Qobj([[np.sqrt(p2),0],[0,np.sqrt(p2)]])
        def sq(r, ops, q):
            rn = r.full(); res = np.zeros_like(rn,dtype=complex)
            for K in ops:
                k = K.full()
                fK = np.kron(k,np.eye(2)) if q==0 else np.kron(np.eye(2),k)
                res += fK @ rn @ fK.conj().T
            return Qobj(res, dims=[[2,2],[2,2]])
        rho = sq(rho,[K0,K1],0); rho = sq(rho,[K0,K1],1)
        rho = sq(rho,[D0,D1],0); rho = sq(rho,[D0,D1],1)
        return rho


def pol_drift(rho, d_km):
    def su2():
        th = RNG.normal(0, POLARIZ_DRIFT_SD*np.sqrt(d_km))
        n = RNG.standard_normal(3); n /= np.linalg.norm(n)+1e-12
        return expm(-1j*(th/2)*(
            n[0]*np.array([[0,1],[1,0]])+
            n[1]*np.array([[0,-1j],[1j,0]])+
            n[2]*np.array([[1,0],[0,-1]])))
    U = Qobj(np.kron(su2(),su2()), dims=[[2,2],[2,2]])
    return U*rho*U.dag()


def phase_noise(rho, d_km):
    tau = d_km*1e3/SPEED_OF_LIGHT
    phi = RNG.normal(0, np.sqrt(2*np.pi*LASER_LINEWIDTH*tau))
    Rz = np.array([[np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]])
    U = Qobj(np.kron(Rz,Rz), dims=[[2,2],[2,2]])
    return U*rho*U.dag()


def jitter_vis():
    return float(erf(COINC_WINDOW_NS*1e-9/(np.sqrt(2)*TIMING_JITTER_PS*1e-12)))


def transmittance(d_km, alt=KATHMANDU_ALT):
    den = np.exp(-alt/ATM_SCALE_HEIGHT)
    T_atm = 10**(-ATM_LOSS_SEA*den*d_km/10)
    div = WAVELENGTH/0.10
    spot = div*d_km*1e3
    T_geo = min(1.0, (0.05/spot)**2)
    return T_atm*T_geo


def mobile_extra_loss(speed_kmh, d_km):
    doppler_factor = 1 - speed_kmh/(3e5)
    pointing_err = np.exp(-0.01 * speed_kmh / max(d_km, 0.01))
    handover_prob = min(0.3, speed_kmh * 0.003)
    return doppler_factor * pointing_err * (1 - handover_prob * 0.5)


# Ref: Nielsen & Chuang, Quantum Computation and Quantum Information, Ch. 8
# Ref: Taylor et al., Phys. Rev. A 110, 022622 (2024)
def loss_ch(rho, T):
    T_per_photon = np.sqrt(max(T, 0.0))
    p_loss = 1.0 - T_per_photon

    K0 = Qobj([[1, 0], [0, T_per_photon]])
    K1 = Qobj([[0, p_loss**0.5], [0, 0]])

    def _apply_single_qubit(r, ops, q):
        rn = r.full(); res = np.zeros_like(rn, dtype=complex)
        for K in ops:
            k = K.full()
            fK = np.kron(k, np.eye(2)) if q==0 else np.kron(np.eye(2), k)
            res += fK @ rn @ fK.conj().T
        return Qobj(res, dims=[[2,2],[2,2]])

    rho = _apply_single_qubit(rho, [K0, K1], 0)
    rho = _apply_single_qubit(rho, [K0, K1], 1)
    return rho
