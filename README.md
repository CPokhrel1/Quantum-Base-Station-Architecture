# Quantum Base Station (QBS)-Enabled Wireless Entanglement Distribution in Kathmandu

A simulation framework for investigating **Quantum Base Station (QBS)-enabled wireless entanglement distribution** in an urban cellular network environment, using Kathmandu Valley, Nepal, as a geographical case study.

The framework integrates free-space optical (FSO) quantum channel modelling, entanglement generation, mobile-node connectivity, entanglement purification, entanglement swapping, and fidelity-aware routing to investigate the feasibility and performance of next-generation wireless quantum networks.

---

## Overview

The emergence of quantum communication and quantum networking introduces the possibility of distributing entanglement across geographically distributed network nodes. Integrating quantum communication capabilities into existing wireless infrastructure provides a potential pathway toward scalable urban quantum networks.

This project investigates a **Quantum Base Station (QBS)-enabled network architecture** in which geographically distributed base stations facilitate entanglement distribution between infrastructure nodes and mobile users.

Kathmandu Valley is considered as the geographical case study. Cellular base-station locations are derived from a geographical cellular-tower dataset, while mobile users are synthetically generated to represent different mobility categories.

The simulation evaluates quantum-channel characteristics, entanglement quality, network connectivity, purification performance, and end-to-end entanglement distribution through multi-hop routing.

## Research Objectives

The primary objectives of this simulation are:

- Develop a QBS-enabled architecture for wireless quantum entanglement distribution.
- Investigate the feasibility of integrating quantum communication with existing cellular infrastructure.
- Model free-space optical quantum communication links between base stations and mobile users.
- Evaluate entanglement fidelity and quantum communication performance under channel impairments.
- Investigate entanglement purification and swapping for improving end-to-end quantum-state quality.
- Develop fidelity-aware routing mechanisms for multi-hop quantum communication.
- Analyse network performance under different mobile-node categories and communication conditions.

---

## Network Architecture

The proposed architecture consists of three principal components:

### 1. Quantum Base Stations

Existing cellular infrastructure is used as the geographical foundation for constructing candidate quantum base-station locations.

The cellular-tower dataset is geographically filtered to the Kathmandu Valley region. DBSCAN clustering is applied to aggregate nearby tower records into representative base-station sites.

The resulting network is represented as a graph:

- Nodes represent quantum base-station sites.
- Edges represent candidate wireless quantum communication links.
- Edge attributes represent simulated quantum-link characteristics.

### 2. Wireless Quantum Communication Links

Free-space optical communication is considered for distributing photonic quantum states between geographically separated nodes.

The simulation incorporates distance-dependent optical transmission and selected channel impairments, including:

- Atmospheric attenuation.
- Polarization drift.
- Phase noise.
- Timing jitter.
- Optical coupling efficiency.
- Mobile-node-related channel effects.

### 3. Mobile Quantum Connectivity

Mobile nodes are introduced to investigate quantum connectivity between users and nearby quantum base stations.

Three mobile categories are considered:

| Mobile Category | Population Distribution | Speed Range |
|---|---:|---:|
| Pedestrian | 50% | 0–6 km/h |
| Vehicle | 35% | 0–80 km/h |
| IoT | 15% | 0–2 km/h |

Mobile nodes are synthetically generated within the geographical study area and associated with nearby base stations according to the configured association distance.

---

## Simulation Framework

The simulation consists of the following major processing stages:

**1. Geographical Network Construction**

Cellular-tower records are processed and clustered to identify representative base-station locations.

**2. Network Connectivity Modelling**

Candidate BTS-to-BTS links are established based on geographical separation and the maximum communication distance.

**3. Mobile-Node Generation**

Pedestrian, vehicle, and IoT nodes are generated with different mobility characteristics.

**4. Quantum Entanglement Generation**

A heralded entanglement-generation procedure is used to model the generation of quantum-correlated states.

**5. Quantum Channel Evaluation**

The generated quantum states are evaluated under the configured FSO channel and noise models.

**6. Entanglement Purification**

BBPSSW and DEJMPS-based purification procedures are implemented to investigate improvements in entanglement fidelity.

**7. Entanglement Swapping**

Bell-state measurement-based entanglement swapping is modelled to establish end-to-end entanglement across multiple network hops.

**8. Fidelity-Aware Routing**

A Dijkstra-based routing mechanism is implemented using fidelity-derived link costs to identify suitable multi-hop quantum communication paths.

**9. Performance Evaluation**

Network-level and quantum-level performance metrics are calculated and visualized.

---

## Quantum Communication and Performance Metrics

The framework evaluates several metrics associated with quantum-state quality, communication performance, and network connectivity.

### Quantum-State Metrics

| Metric | Description |
|---|---|
| Fidelity | Measures the quality of the generated quantum state relative to the target state. |
| QBER | Estimates the quantum bit error rate. |
| Concurrence | Quantifies bipartite entanglement. |
| Logarithmic Negativity | Provides an additional measure of quantum entanglement. |

### Communication and Network Metrics

- Entanglement generation rate.
- Secret key rate (SKR) estimates.
- Quantum-channel transmittance.
- Communication latency.
- Decoherence and coherence-time indicators.
- Optical link-budget characteristics.
- End-to-end entanglement fidelity.
- Purification performance.
- Network connectivity and routing characteristics.
- Mobile-node connectivity and base-station loading.

The metrics are calculated using the models implemented in the respective simulation modules.

---

## Simulation Configuration

The principal simulation parameters are defined in `quantum_bts/config.py`.

| Parameter | Configuration |
|---|---:|
| Optical wavelength | 810 nm |
| Kathmandu reference altitude | 1,350 m |
| Atmospheric scale height | 8,500 m |
| Maximum BTS-to-BTS link distance | 12 km |
| Maximum mobile association distance | 4.5 km |
| Number of mobile nodes | 100 |
| Monte Carlo trials | 80 |
| Timing jitter | 50 ps |
| Laser linewidth | 1 kHz |
| Detector efficiency | 0.85 |
| Dark count rate | 200 Hz |
| Photon emission probability | 0.03 |
| Source repetition rate | 80 MHz |
| Maximum purification rounds | 3 |
| Random seed | 42 |

These values represent the current simulation configuration and can be modified to investigate alternative network conditions.

---

## Repository Structure

```text
quantum_bts_kathmandu/
│
├── main.py
├── requirements.txt
│
├── quantum_bts/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── channel.py
│   ├── generation.py
│   ├── quantum_metrics.py
│   ├── network.py
│   ├── link_metrics.py
│   ├── purification.py
│   ├── swapping.py
│   ├── routing.py
│   ├── stats.py
│   │
│   └── figures/
│       ├── __init__.py
│       ├── style.py
│       ├── dashboards.py
│       ├── e2e.py
│       ├── extended.py
│       ├── mobile.py
│       ├── network.py
│       ├── noise.py
│       ├── routing.py
│       └── stats.py
│
└── outputs/
```

### Module Description

| Module | Functionality |
|---|---|
| `config.py` | Centralized simulation parameters and configuration. |
| `data.py` | Geographical dataset processing, clustering, and mobile-node generation. |
| `channel.py` | FSO channel modelling and channel impairments. |
| `generation.py` | Heralded quantum entanglement generation. |
| `quantum_metrics.py` | Quantum-state fidelity and entanglement metrics. |
| `network.py` | BTS graph construction and mobile-link simulation. |
| `link_metrics.py` | Extended communication and link-performance metrics. |
| `purification.py` | BBPSSW and DEJMPS-based entanglement purification. |
| `swapping.py` | Bell-state measurement-based entanglement swapping. |
| `routing.py` | Fidelity-aware Dijkstra routing and end-to-end evaluation. |
| `stats.py` | Statistical analysis of simulation results. |
| `figures/` | Network, quantum-channel, mobility, routing, and performance visualizations. |

---

## Installation

### Prerequisites

- Python 3.x
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/CPokhrel1/Quantum-Base-Station-Architecture
cd quantum_bts_kathmandu
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate the environment:

**Linux/macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The principal dependencies are:

- NumPy
- Pandas
- Matplotlib
- NetworkX
- SciPy
- scikit-learn
- QuTiP

---

## Input Dataset

The simulation requires a cellular-tower dataset named:

```text
429.csv
```

The dataset is loaded by `quantum_bts/data.py`.

The geographical filtering uses the following approximate bounding coordinates:

| Coordinate | Minimum | Maximum |
|---|---:|---:|
| Latitude | 27.60° | 27.80° |
| Longitude | 85.25° | 85.45° |

DBSCAN clustering is configured with:

- Epsilon (`eps`): 0.004
- Minimum samples (`min_samples`): 1
- Distance metric: Euclidean

**Note:** The input dataset is not included in this repository. Users must obtain the appropriate dataset and place `429.csv` in the repository root before executing the simulation. Please verify the dataset's licence and redistribution permissions independently.

---

## Running the Simulation

After installing the dependencies and providing the required dataset, execute:

```bash
python main.py
```

The main simulation script performs the following operations:

1. Loads and processes cellular-tower records.
2. Constructs representative QBS locations.
3. Generates mobile nodes.
4. Builds the base-station connectivity graph.
5. Establishes mobile-to-base-station associations.
6. Simulates wireless quantum links.
7. Evaluates quantum-channel and communication metrics.
8. Performs entanglement purification.
9. Evaluates entanglement swapping and multi-hop routing.
10. Generates statistical summaries and visualization outputs.

Generated figures and results are stored in the configured output directory.

---

## Visualization and Results

The framework includes visualization modules for analysing different aspects of the simulated quantum network.

The generated visualizations cover:

- Geographical distribution of quantum base stations.
- BTS-to-BTS network topology.
- Mobile-node spatial distribution.
- Mobile-node coverage and connectivity.
- Quantum-state fidelity distribution.
- Fidelity heatmaps.
- QBER and transmittance analysis.
- Channel noise breakdown.
- Mobile-node fidelity versus distance and speed.
- Pedestrian, vehicle, and IoT performance comparisons.
- Base-station load analysis.
- Entanglement purification performance.
- Secret key rate analysis.
- Quantum communication latency.
- Link-budget analysis.
- End-to-end entanglement performance.
- Fidelity-aware routing and network statistics.

---

## Reproducibility

A fixed random-number generator seed is configured in the simulation to support reproducibility of stochastic experiments.

For consistent results, retain the following information when conducting experiments:

- Simulation configuration.
- Input dataset and its version.
- Python and dependency versions.
- Random-number seed.
- Experimental output files.

Changes to these parameters may affect the simulation results.

---

## Research Scope and Limitations

This repository provides a simulation-based investigation of QBS-enabled wireless quantum networking.

The results should be interpreted within the assumptions and abstractions of the implemented models.

In particular:

- Mobile-node positions and mobility characteristics are synthetically generated.
- The FSO channel represents a model of wireless optical quantum communication, not a field measurement.
- Hardware and channel parameters are simulation inputs.
- Communication and secret-key-rate metrics are model-derived estimates.
- Candidate graph links do not imply the existence of deployed physical quantum links.
- The framework does not constitute an experimental demonstration using deployed quantum hardware.

Further investigation may include detailed atmospheric turbulence and weather modelling, experimentally calibrated channel parameters, quantum-memory characterization, and hardware-based validation.

---

## Citation

If you use this repository or build upon its implementation in academic research, please cite the associated research publication or manuscript when available.

**Software citation:**

```text
Pokhrel, C. (2026).
Quantum Base Station (QBS)-Enabled Wireless Entanglement
Distribution in Kathmandu.
GitHub repository: https://github.com/CPokhrel1/Quantum-Base-Station-Architecture
```

---

## License

This project is distributed under the MIT License.

See the `LICENSE` file for details.

---

## Acknowledgements

This project uses open-source scientific computing and quantum simulation libraries, including NumPy, Pandas, SciPy, NetworkX, scikit-learn, Matplotlib, and QuTiP.

Their respective authors and contributors are acknowledged for providing the tools that support this research.
