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

## 4. Guidelines for AI Coding Agents

1. **Verify Source Code**: Make changes in `src/aqnet/` or `tests/`. Do not edit build artifacts or temporary outputs.
2. **Always Run Tests**: After making changes, run `ruff check .` and `pytest --cov=aqnet --cov-report=term-missing tests/` to verify that all tests pass and linting is clean.
3. **Preserve Mathematical Invariants**: Ensure analytical formulas match the renewal-reward theory and fixed-point traffic conservation formulas defined in the research documentation and research paper.
