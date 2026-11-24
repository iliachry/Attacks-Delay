# Changelog

All notable changes to `aqnet` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-24

First release intended for PyPI. Packages the saddle-node retry-bifurcation research from
0.2.0 as a documented public API with tutorials.

### Added
- `aqnet.fold_parameter_v(mu, T, method="wrightomega")`: the fold parameter
  `v = ω(μT + 1) = W₀(e^(μT+1))`, computed with Wright-omega (default) or Lambert-W₀ for
  cross-checking. The Lambert-W path raises `ValueError` instead of overflowing when `μT > 708`.
- `aqnet.TopologyName`: `typing.Literal` of the five supported topologies, now used in the
  signatures of the bifurcation functions.
- API reference: `docs/api.md`.
- Three executed tutorial notebooks in `notebooks/`: basic networks (theory vs SimPy for
  all five topologies), attack/timeout retry storms (including burst-triggered collapse
  probability), and the saddle-node bifurcation (closed form, branches, square-root law,
  critical-capacity map, SymPy proofs).
- `scripts/build_notebooks.py`, which regenerates the notebooks from source.
- `notebooks` optional extra (`nbconvert`, `nbformat`, `ipykernel`).
- `tests/test_saddle_node_closed_form.py` (178 tests) checking every closed form against
  independent SciPy root-finding: fold location, critical capacity, curvature, Case 1
  `p`-invariance, Case 2 `p_crit`, both fixed-point branches, square-root normal-form
  convergence, multi-node folds, and the docstring examples.
- `CHANGELOG.md` and a `LICENSE` file (MIT text, now shipped in the sdist and wheel).

### Changed
- Rewrote the docstrings in `aqnet.models.bifurcation` in NumPy style: the module docstring
  states the theorem and closed forms, and every public function and result field is
  documented with parameters, returns, raises and runnable examples.
- `solve_saddle_node_bifurcation` now computes the closed-form fold through
  `fold_parameter_v` (numerically identical).
- README: saddle-node usage section, links to the API reference and notebooks, and a live
  PyPI version badge in place of a hard-coded one.

## [0.2.0] - 2026-11-12

Not published to PyPI.

### Added
- Saddle-node (fold) retry bifurcation module `aqnet.models.bifurcation`: closed-form
  Wright-omega fold for Cases 1–2, numerical folds for Cases 3–5, dual fixed-point solver,
  SymPy proofs, and SimPy metastable-collapse simulation.
- `aqnet bifurcation` CLI subcommand.
- Two-station edge-cloud confidence cascade (`aqnet.models.cascade`,
  `aqnet.models.cascade_nonstationary`) with non-stationary envelope, implicit-function
  sensitivity and quartic hysteresis.
- Interactive web studio (`web/`), LaTeX manuscripts and publication figures.

### Changed
- Unified the five topology solvers around the shared throughput-conservation map.

## [0.1.0] - 2026-08-28

Not published to PyPI.

### Added
- Installable `aqnet` package: analytical solvers and SimPy simulators for single-node
  destruction and modification, tandem chain, feedforward and feedback networks.
- `aqnet run` and `aqnet bench` CLI.

[0.3.0]: https://github.com/iliachry/Attacks-Delay/compare/53f457d...v0.3.0
[0.2.0]: https://github.com/iliachry/Attacks-Delay/compare/a84ee4e...53f457d
[0.1.0]: https://github.com/iliachry/Attacks-Delay/commit/a84ee4e
