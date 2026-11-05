import numpy as np
import qutip as qt
from qutip import bell_state, concurrence, fidelity, ket2dm


_H2 = np.array([[1,1],[1,-1]])/np.sqrt(2)
_H2_KRON = np.kron(_H2, _H2)


# Ref: Bennett & Brassard (1984); Ekert (1991); Shor & Preskill (2000)
# Ref: Gisin et al., Rev. Mod. Phys. 74, 145 (2002), Sec. IV.A
def qber_from_rho(rho):
    rho_np = rho.full()
    e_Z = float(np.real(rho_np[1,1])) + float(np.real(rho_np[2,2]))

    rho_X = _H2_KRON @ rho_np @ _H2_KRON.conj().T
    e_X = float(np.real(rho_X[1,1])) + float(np.real(rho_X[2,2]))

    return max(e_Z, 0.0), max(e_X, 0.0)


def metrics(rho):
    phi = bell_state("00"); target = ket2dm(phi)
    F = float(fidelity(rho, target)**2)
    try: C = float(concurrence(rho))
    except: C = 0.0
    e_Z, e_X = qber_from_rho(rho)
    Q = max(e_Z, e_X)
    try: N = float(qt.logarithmic_negativity(rho))
    except: N = 0.0
    return F, C, Q, N
