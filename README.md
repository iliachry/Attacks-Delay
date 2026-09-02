# aqnet — Adversarial Queue Networks

[![PyPI - Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/aqnet/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-cs.NI%2Fmath.PR-b31b1b.svg)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-5%20passed-brightgreen.svg)]()

> **Mathematical Modeling and Performance Analysis of Multi-Node Queueing Networks Under Adversarial Attacks and Timeout-Driven Retransmissions**  
> **Author:** Ilias Chrysovergis (Imperial College London & Metatopia)  
> **Preprint / Target:** IEEE Transactions on Networking (ToN) / arXiv (`cs.NI`, `math.PR`, `cs.CR`)

---

## ⚡ Quickstart

### Installation

Install via pip:

```bash
pip install aqnet
```

Or install in development mode from source:

```bash
git clone https://github.com/iliachry/Attacks-Delay.git
cd Attacks-Delay
pip install -e .
```

### Python API Usage

```python
import aqnet

# 1. Closed-form analytical delay calculation (Case 1: Pre-service Destruction)
theory_delay = aqnet.solve_one_node_destruction(
    lambda_n=2.0,  # Fresh packet arrival rate
    p=0.2,         # Attack probability
    mu=10.0,       # Server capacity
    T=2.0          # Retransmission timeout window
)
print(f"Expected Sojourn Time: {theory_delay:.4f} s")

# 2. Run SimPy discrete-event simulation
sim_delay = aqnet.simulate_one_node_destruction(
    lambda_n=2.0,
    p=0.2,
    mu=10.0,
    T=2.0,
    sim_duration=5000.0,
    seed=42
)
print(f"Simulated Sojourn Time: {sim_delay:.4f} s")

# 3. Analyze multi-hop tandem chain (Case 3)
tandem_delay = aqnet.solve_tandem_theory(p=0.1, N=3, mu=2.0, lambda_arrival=0.15, W=8.0)
print(f"Tandem (N=3) Delay: {tandem_delay:.4f} s")
```

### Command Line Interface (CLI)

`aqnet` comes with a built-in CLI for rapid experimentation and benchmarking:

```bash
# Run benchmark across all 5 topologies:
aqnet bench

# Simulate specific topology:
aqnet run --topology tandem --nodes 3 --p 0.1 --lambda-arr 0.15 --reps 20
```

---

## Overview

The reliable transport of data through adversarial networks requires a rigorous mathematical understanding of the coupled dynamics between malicious disruption and transport-layer recovery protocols. 

This repository and package provides the complete analytical framework, discrete-event simulation engine (SimPy), benchmark dataset, and LaTeX manuscript files for studying packet sojourn times across **five foundational network topologies** subject to active packet destruction and modification attacks.

By synthesizing classical queueing theory, renewal-reward theory, and fixed-point traffic conservation, this project validates closed-form and semi-analytical models against discrete-event simulations, demonstrating exact agreement across all operational regimes with relative errors consistently below 1%.

---

## Key Topologies & Case Studies

| Topology / Case Study | Attack Model | Analytical Tool | Key Dynamic | Script Location |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1: 1-Node Destruction** | Pre-service packet destruction | Renewal-Reward & Tail Probabilities | Packet destroyed before service; timeout/backoff trigger retransmissions | [`1_one_node_destruction/one_node_packet_attack.py`](./1_one_node_destruction/one_node_packet_attack.py) |
| **Case 2: 1-Node Modification** | Post-service payload corruption | Fixed-Point Traffic Conservation | Packet completes service; corruption detected at receiver; full server cycle wasted per attempt | [`2_one_node_modification/one_node_packet_modification.py`](./2_one_node_modification/one_node_packet_modification.py) |
| **Case 3: Tandem Chain** | Multi-hop stage-wise attacks | Hypoexponential Sojourn Matching | Multi-server journey times with cumulative attack probabilities | [`3_tandem_chain/tandem.py`](./3_tandem_chain/tandem.py) |
| **Case 4: Feedforward Network** | $N$-node pipeline attacks | Gamma Moment Matching & Truncated Expectations | Non-linear traffic amplification loops, stage-wise attenuation, and stability envelope contraction | [`4_n_node_feedforward/n_node_feedforward.py`](./4_n_node_feedforward/n_node_feedforward.py) |
| **Case 5: Feedback Mesh** | $N$-node symmetric feedback | Erlang Multi-Visit Path Traversal | Distributed mesh routing prevents single-node bottlenecking | [`5_n_node_feedback/n_node_with_feedback.py`](./5_n_node_feedback/n_node_with_feedback.py) |

---

## Repository Structure

```text
Attacks-Delay/
├── 1_one_node_destruction/        # Case 1: Single-node pre-service destruction
│   ├── one_node_packet_attack.py
│   ├── results_destruction.json
│   └── destroy_no_service_plot_reps200.png
├── 2_one_node_modification/       # Case 2: Single-node post-service modification
│   ├── one_node_packet_modification.py
│   ├── results_modification.json
│   └── plot_reps50_warmup500_sim5000.png
├── 3_tandem_chain/                # Case 3: Multi-hop tandem chain
│   ├── tandem.py
│   └── corrected_tandem_simulation_N3.png
├── 4_n_node_feedforward/          # Case 4: General N-node feedforward network
│   ├── n_node_feedforward.py
│   ├── results_feedforward.json
│   ├── section_3_3_2_stability_regions.png
│   └── section_3_3_2_tandem_delay_vs_N_varying_p.png
├── 5_n_node_feedback/             # Case 5: N-node symmetric feedback mesh
│   ├── n_node_with_feedback.py
│   ├── results_feedback.json
│   └── section_3_3_1_sojourn_vs_N_varying_p.png
├── arxiv_package/                 # Standalone arXiv submission bundle (main.tex + figures)
├── letter/                        # Cover letters and editorial correspondence
│   ├── cover_letter.tex
│   └── letter.tex
├── paper_ieee.tex                 # IEEE Transactions two-column manuscript
├── paper.tex                      # Standard single-column full research manuscript
├── ARXIV_METADATA.txt             # arXiv submission metadata (Title, Abstract, Categories)
├── SUBMISSION_GUIDE.md            # Step-by-step submission guide for arXiv and IEEE ToN
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.8+**
- **TeX Live / MacTeX** (optional, required only for compiling LaTeX paper manuscripts)

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/iliachry/Attacks-Delay.git
cd Attacks-Delay

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Experiments & Generating Plots

Each case study can be executed independently. The scripts automatically compute theoretical analytical values, execute Monte Carlo SimPy discrete-event simulations across parameter sweeps, verify stability boundaries, and generate publication-quality figures:

### Case 1: Single-Node Pre-Service Destruction
```bash
cd 1_one_node_destruction
python one_node_packet_attack.py
```
*Outputs:* `results_destruction.json`, `destroy_no_service_plot_reps200.png`

### Case 2: Single-Node Post-Service Modification
```bash
cd 2_one_node_modification
python one_node_packet_modification.py
```
*Outputs:* `results_modification.json`, `plot_reps50_warmup500_sim5000.png`

### Case 3: Tandem Queueing Chain
```bash
cd 3_tandem_chain
python tandem.py
```
*Outputs:* `corrected_tandem_simulation_N3.png`

### Case 4: $N$-Node Feedforward Network
```bash
cd 4_n_node_feedforward
python n_node_feedforward.py
```
*Outputs:* `results_feedforward.json`, `section_3_3_2_tandem_delay_vs_N_varying_p.png`, `section_3_3_2_stability_regions.png`, `section_3_3_2_tandem_throughput_vs_N.png`

### Case 5: $N$-Node Symmetric Feedback Mesh
```bash
cd 5_n_node_feedback
python n_node_with_feedback.py
```
*Outputs:* `results_feedback.json`, `section_3_3_1_sojourn_vs_N_varying_p.png`

---

## Compiling Manuscripts & Publication Materials

### Compiling LaTeX Papers
```bash
# Compile IEEE Transactions format (two-column)
pdflatex -interaction=nonstopmode paper_ieee.tex
pdflatex -interaction=nonstopmode paper_ieee.tex

# Compile Full Manuscript (single-column)
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

### Packaging for arXiv Submission
The self-contained arXiv submission bundle can be generated directly:
```bash
cd arxiv_package
tar -czvf ../arxiv_submission.tar.gz main.tex *.png
```
See [`SUBMISSION_GUIDE.md`](./SUBMISSION_GUIDE.md) and [`ARXIV_METADATA.txt`](./ARXIV_METADATA.txt) for submission details.

---

## Citation

If you find this codebase or theoretical framework helpful in your research, please cite:

```bibtex
@article{chrysovergis2026adversarial,
  title   = {Mathematical Modeling and Performance Analysis of Multi-Node Queueing Networks Under Adversarial Attacks and Timeout-Driven Retransmissions},
  author  = {Chrysovergis, Ilias},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
