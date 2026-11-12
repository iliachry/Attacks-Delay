# Agent Guidelines for aqnet

This document provides guidelines, architectural details, coding standards, and commands for AI coding agents and human contributors working on `aqnet`.

---

## 1. Project Overview & Architecture

`aqnet` (Adversarial Queue Networks) is a Python library and CLI tool for mathematical modeling and discrete-event simulation of multi-node queueing networks under adversarial packet destruction/modification attacks and timeout-driven retransmissions.

### Core Architecture (`src/aqnet/`)

- **`src/aqnet/models/`**: Analytical solvers for queueing network topologies:
  - `one_node.py`: Analytical models for single-node queue with pre-service destruction (Case 1: `solve_one_node_destruction`) and post-service modification (Case 2: `solve_one_node_modification`).
  - `tandem.py`: Multi-hop tandem chain solver (Case 3: `solve_tandem_theory`).
  - `feedforward.py`: $N$-node feedforward network solver (Case 4: `solve_feedforward_theory`).
  - `feedback.py`: $N$-node symmetric feedback mesh solver (Case 5: `solve_feedback_theory`).
- **`src/aqnet/simulation/`**: SimPy discrete-event simulation engines:
  - `engine.py`: SimPy simulation routines for all 5 case studies (`simulate_one_node_destruction`, `simulate_one_node_modification`, `simulate_tandem`, `simulate_feedforward`, `simulate_feedback`).
- **`src/aqnet/cli.py`**: Command-Line Interface (`aqnet run`, `aqnet bench`).
- **`tests/`**: Pytest test suite for theoretical models and simulations.
- **Top-level directory case study folders**: (`1_one_node_destruction/`, `2_one_node_modification/`, `3_tandem_chain/`, `4_n_node_feedforward/`, `5_n_node_feedback/`) containing stand-alone reproduction scripts and benchmark outputs.

---

## 2. Coding Standards & Conventions

- **Python Version**: Compatibility with Python 3.8+.
- **Type Annotations**: Use Python type hints on all public function signatures.
- **Formatting & Linting**: Adhere strictly to PEP 8 standards enforced via `ruff`.
- **Numerical Safety**: Always check for numerical stability boundaries (e.g., traffic intensity $\rho \ge 1$, division by zero, non-convergent iterations) and return `float("inf")`, `None`, or raise appropriate errors as expected by the solver interfaces.
- **Testing**: Maintain comprehensive pytest coverage across all modules, including edge cases ($p=0$, high attack rates, boundary stability conditions, invalid inputs).

---

## 3. Local Development, Linting & Testing Commands

Before committing or submitting changes, ensure you run all checks locally:

```bash
# Install package in editable mode with development dependencies
pip install -e .[dev]

# Lint code using ruff
ruff check .

# Run pytest test suite with coverage
pytest --cov=aqnet --cov-report=term-missing tests/
```

---

## 4. Master Feature Catalog (`FEAT-01`..`FEAT-08` — `aqnet v0.2.0`)

- **`FEAT-01` — Five-Topology Adversarial Queueing & Saddle-Node Bifurcation Switcher**: Interactive topology selector across Case 1 (`one_node_destruction`), Case 2 (`one_node_modification`), Case 3 (`tandem`), Case 4 (`feedforward`), and Case 5 (`feedback`), plus the Two-Station Confidence Cascade (`cascade`).
- **`FEAT-02` — Closed-Form Wright-$\omega$ / Lambert-$W_0$ Saddle-Node Fold & Dual-Branch Solver**: Real-time calculation of $(\Lambda_{\text{SN}}^*, \lambda_{\text{crit}}^{\text{SN}}, \rho_{\text{SN}}, c_2)$, stable lower fixed-point branch $\Lambda_{\text{stable}}^*(\lambda)$, and unstable upper separatrix $\Lambda_{\text{unstable}}^*(\lambda)$ with square-root normal-form overlay.
- **`FEAT-03` — Dual-Canvas Phase Portrait & Critical Capacity Manifold (`<canvas>`)**: Interactive high-DPI phase canvas rendering either the unimodal throughput conservation curve $\Gamma(\Lambda^*)$ with dual fixed-point markers or the critical capacity manifold $\lambda_{\text{crit}}^{\text{SN}}(p)$ contrasting Case 1 structural $p$-invariance against Case 2 linear decay.
- **`FEAT-04` — Law `L10` Generative Friction Offering (Adversarial Burst & Custom Workload Trace Probe)**: Interactive packet/burst offering (`#burst-input`, `#btn-inject-burst`, and preset scenario chips) that injects an empirical arrival-burst trace into the discrete-event queue and evaluates whether the trajectory stays inside the stable attraction basin or crosses the unstable separatrix $\Lambda_{\text{unstable}}^*$ into runaway retry collapse.
- **`FEAT-05` — Two-Station Edge-Cloud Confidence Cascade & Quartic Hysteresis Inspector**: Interactive $(\tau^*, b, T, \Delta\tau_{\text{hyst}}^*)$ renewal-reward Pareto frontier, Jensen micro-batch antifragility bound ($\mathcal{C}_{\text{batch}}(b,T) < 0$ for $b \in \{4,8\}$), and quartic $O((\sigma_\lambda^2 \tau_\lambda)^{1/4})$ deadband calculator.
- **`FEAT-06` — SymPy Proof Ledger & LaTeX / BibTeX Citation Modal (`<dialog>`)**: Progressive-disclosure `<dialog>` drawer (`#proof-dialog`) exposing the 4 machine-verified SymPy identities, closed-form Wright-$\omega$ derivations, JSON telemetry export, and one-click BibTeX citation copy.
- **`FEAT-07` — Chameleon 3-Theme Polarity Engine (`academic-paper`, `metatopia-light`, `cyber-gold`)**: Instant theme switching (`data-theme`) with tabular numerals (`font-variant-numeric: tabular-nums`), strict kanon regularity (`R_kanon = 1.0`), and single-row `<= 56px` header chrome (`Laws L0–L8`).
- **`FEAT-08` — Reproducible CLI & Quality Gate (`aqnet run`, `aqnet bifurcation`, `aqnet bench`)**: Deterministic Python package (`src/aqnet/`) and headless verification pipeline (`scripts/quality_gate.sh`).

---

## 5. Guidelines for AI Coding Agents

1. **Verify Source Code**: Make changes in `src/aqnet/`, `web/`, or `tests/`. Do not edit build artifacts or temporary outputs.
2. **Always Run Tests**: After making changes, run `uvx ruff check src/ tests/` and `uv run pytest` to verify that all tests pass and linting is clean.
3. **Preserve Mathematical Invariants**: Ensure analytical formulas match the renewal-reward theory, Wright-$\omega$ Saddle-Node bifurcation theorem, and fixed-point traffic conservation formulas defined in the research paper.

## Activated Sovereign Agent Skills (`kb skills --archetype python-uv-ai`)

| Skill ID | Local Entry Point (`SKILL.md` + `references/`) |
| :--- | :--- |
| `goldwood-gwi` | `~/.gemini/config/skills/goldwood-gwi/SKILL.md` |
| `gx10-cluster` | `~/.gemini/config/skills/gx10-cluster/SKILL.md` |
| `stochastic-math-econometrics` | `~/.gemini/config/skills/stochastic-math-econometrics/SKILL.md` |
| `web3-fintech-settlement` | `~/.gemini/config/skills/web3-fintech-settlement/SKILL.md` |
| `huggingface-ml` | `~/.gemini/config/skills/huggingface-ml/SKILL.md` |
| `mcp-builder` | `~/.gemini/config/skills/mcp-builder/SKILL.md` |
| `supabase-postgres` | `~/.gemini/config/skills/supabase-postgres/SKILL.md` |
| `principal-engineering` | `~/.gemini/config/skills/principal-engineering/SKILL.md` |
