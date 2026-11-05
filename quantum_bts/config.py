import os

import numpy as np

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

SPEED_OF_LIGHT = 3e8
WAVELENGTH = 810e-9
ATM_SCALE_HEIGHT = 8500
ATM_LOSS_SEA = 0.20
KATHMANDU_ALT = 1350
TIMING_JITTER_PS = 50
LASER_LINEWIDTH = 1e3
POLARIZ_DRIFT_SD = 0.05
N_MC = 80
COINC_WINDOW_NS = 1.0
MAX_LINK_KM = 12.0
MAX_MOBILE_KM = 4.5
N_MOBILE = 100
RNG = np.random.default_rng(42)

# SNSPD
DETECTOR_EFFICIENCY = 0.85
DARK_COUNT_RATE = 200  # Hz per detector

# Ref: Evans et al., Science 362, 662 (2018)
# Ref: Nguyen et al., PRX Quantum 4, 010307 (2023)
SOURCE_REP_RATE_HZ = 8.0e7
PHOTON_EMISSION_PROB = 0.03

# Ref: Walmsley group, Nature 530, 313 (2016)
BSM_HARDWARE_MS = 0.001

# Ref: Lutkenhaus, Calsamiglia & Suominen, Phys. Rev. A 59, 3295 (1999)
BSM_LINEAR_OPTIC_CAP = 0.5

CLASSICAL_PROC_MS = 0.002

# Ref: Bradley et al., Phys. Rev. X 9, 031045 (2019)
FEEDFORWARD_MS = 0.005

COHERENCE_BASE_MS = 3.0
COHERENCE_ATM_DECAY = 0.12  # ms/km

# Ref: Dikmelik & Davidson, Appl. Opt. 44, 4946 (2005)
# Ref: Weyrauch & Vorontsov, Appl. Opt. 43, 4353 (2004)
FSO_COUPLING_EFF = 0.05

SOURCE_PAIR_RATE_HZ = SOURCE_REP_RATE_HZ * PHOTON_EMISSION_PROB

HERALD_CAP_MULTIPLE = 25
HERALD_ABS_CAP = 2_000_000
DARK_COUNT_WINDOW_NS = COINC_WINDOW_NS

PURIFICATION_F_LOW = 0.50
PURIFICATION_F_HIGH = 0.85
PURIFICATION_MAX_ROUNDS = 3

# Ref: arXiv:2008.06458 (NV two-qubit gate fidelity)
ETA_SWAP_GATE = 0.992
# Ref: Zhang et al., spin-to-charge conversion readout, PMC7943573
# Ref: entangledfuture.com, NV-Center Quantum Computing survey (2026)
ETA_SWAP_READOUT = 0.955
ETA_SWAP = ETA_SWAP_GATE * ETA_SWAP_READOUT

SWAP_FIDELITY_THRESH = 0.50
QKD_FIDELITY_THRESH = 2 / 3
