# aqnet API reference

Everything listed here is importable from the top-level package (`import aqnet`) and is
covered by the test suite. Rates are in packets per unit time; `T` / `W` is the
retransmission timeout in the same time unit.

- [Topologies](#topologies)
- [Saddle-node retry bifurcation](#saddle-node-retry-bifurcation) (new public API in 0.3.0)
- [Delay solvers](#delay-solvers)
- [Discrete-event simulators](#discrete-event-simulators)
- [Edge-cloud confidence cascade](#edge-cloud-confidence-cascade)
- [Command-line interface](#command-line-interface)

---

## Topologies

`aqnet.TopologyName` is a `typing.Literal` of the five supported topologies:

| Value | Case | Attack model |
| :--- | :--- | :--- |
| `"one_node_destruction"` | 1 | Packet destroyed before service; the server never sees it. |
| `"one_node_modification"` | 2 | Packet served, then found corrupted at the receiver; the service is wasted. |
| `"tandem"` | 3 | `N` servers in series, attack at each hop. |
| `"feedforward"` | 4 | `N`-stage feedforward pipeline with stage-wise attenuation. |
| `"feedback"` | 5 | Symmetric `N`-node mesh with feedback routing. |

---

## Saddle-node retry bifurcation

### The model in one paragraph

With timeout-driven retries the bottleneck attempt rate `Λ` (fresh packets plus retries)
must satisfy the conservation law

```
Γ(Λ) := Λ · P_succ(Λ) = λ
```

where `λ` is the fresh arrival rate and `P_succ(Λ)` the probability that one attempt both
survives the attack and finishes within `T`. `Γ` rises, peaks at the **fold** `Λ_SN` and
falls back to zero at saturation `Λ_max`. For `λ < λ_crit = Γ(Λ_SN)` there are two fixed
points: a stable one and an unstable **separatrix** above it. A burst that pushes the
retry load past the separatrix tips the queue into a retry storm. For `λ > λ_crit` no
fixed point exists and the queue collapses.

### Closed form (Cases 1 and 2)

Let `v` be the real solution of `v·eᵛ = e^(μT+1)`:

```
v = ω(μT + 1) = W₀(e^(μT+1))          (Wright-omega = principal Lambert-W)
```

| Quantity | Case 2 (modification) | Case 1 (destruction) |
| :--- | :--- | :--- |
| `Λ_SN` | `(v − 1) / T` | `(v − 1) / ((1 − p) T)` |
| `λ_crit` | `(1 − p)(v − 1)² / (T v)` | `(v − 1)² / (T v)` (independent of `p`) |
| `Γ''(Λ_SN)` | `−(1 − p) T (1 + v) / v` | `−(1 − p)² T (1 + v) / v` |
| `p_crit(λ)` | `1 − λ / λ_crit(p=0)` | `1` if `λ ≤ λ_crit` else `0` |

Near the fold, both branches follow `Λ± ≈ Λ_SN ± √((λ_crit − λ) / c₂)` with
`c₂ = −Γ''(Λ_SN)/2`, so `dΛ/dλ` diverges like `(λ_crit − λ)^(−1/2)`.

Cases 3–5 have no closed form; the fold is found by bounded Brent maximisation of the
same `Γ`.

### `fold_parameter_v(mu, T, method="wrightomega") -> float`

Returns `v = ω(μT + 1)`. `method="lambertw"` computes `W₀(e^(μT+1))` instead, for
cross-checking; it raises `ValueError` once `μT > 708` because `e^(μT+1)` overflows.

```python
>>> aqnet.fold_parameter_v(10.0, 1.0)
8.82267...
```

### `solve_saddle_node_bifurcation(topology="one_node_modification", p=0.25, mu=10.0, T=1.0, N=2) -> SaddleNodeBifurcationResult`

Locates the fold. Raises `ValueError` if `p ∉ [0, 1)`, `mu ≤ 0`, `T ≤ 0`, `N < 1`, or the
topology is unknown.

`SaddleNodeBifurcationResult` (frozen dataclass):

| Field | Meaning |
| :--- | :--- |
| `topology, p, mu, T, N` | Inputs. |
| `Lambda_max` | Saturation attempt rate (`mu/(1−p)` for Case 1, else `mu`). |
| `Lambda_sn` | Attempt rate at the fold. |
| `lambda_crit_sn` | Largest sustainable fresh arrival rate. |
| `rho_sn` | Bottleneck utilisation at the fold. |
| `p_succ_at_sn` | `lambda_crit_sn / Lambda_sn`. |
| `d2Gamma_dLambda2`, `c2_normal_form` | Curvature at the fold and `c₂ = −Γ''/2`. |
| `naive_capacity`, `capacity_ratio_vs_naive` | Capacity ignoring timeouts, and the fraction that survives retries. |
| `closed_form_exact` | `True` for Cases 1–2. |

Method `p_crit_for_lambda(lambda_ext) -> float` returns the attack probability above which
`lambda_ext` is no longer sustainable (closed form for Cases 1–3, bisection for 4–5).

### `solve_dual_fixed_points(lambda_ext, topology="one_node_modification", p=0.25, mu=10.0, T=1.0, N=2, backoff=0.03) -> DualBranchFixedPointResult`

Solves `Γ(Λ) = lambda_ext` for both branches by Brent's method (tolerance `1e-14`).

| Field | Meaning |
| :--- | :--- |
| `is_stable` | `True` iff `lambda_ext < lambda_crit_sn`. |
| `Lambda_stable`, `Lambda_unstable` | Lower (stable) and upper (separatrix) roots; `nan` when supercritical. |
| `separatrix_gap` | `Lambda_unstable − Lambda_stable`: extra retry load a burst needs to cause collapse. |
| `margin_to_sn` | `lambda_crit_sn − lambda_ext`. |
| `sqrt_asymptotic_stable`, `sqrt_asymptotic_unstable` | Normal-form approximations. |
| `delay_sensitivity_dLambda_dlambda` | `1 / Γ'(Lambda_stable)`. |
| `expected_total_time_stable` | Mean end-to-end sojourn time on the stable branch. |
| `stability_regime` | `"subcritical_bistable"`, `"critical_fold_semistable"` or `"supercritical_collapse"`. |
| `is_critical_fold` | `True` exactly at the fold. |

### Building blocks

- `throughput_curve_gamma(Lambda_star, topology, p, mu, T, N=2) -> float` — `Γ(Λ)`.
- `success_probability_p_succ(Lambda_star, topology, p, mu, T, N=2) -> float` — `P_succ(Λ)`.
- `upper_attempt_rate_bound(topology, p, mu, N=2) -> float` — `Λ_max`.

### Verification

- `verify_symbolic_bifurcation_proofs() -> dict` — SymPy proofs of the stationarity
  identity, the fold curvature, Case 1 `p`-invariance and the square-root normal form.
  Check `result["all_symbolic_proofs_verified"]`.
- `simulate_metastable_retry_collapse_simpy(lambda_ext=1.5, p=0.25, mu=10.0, T=1.0, burst_factor=1.0, burst_duration=0.0, burst_window=None, retry_delay_mean=0.02, max_inflight_cap=250, sim_time=240.0, n_batches=20, seed=42) -> dict`
  — SimPy simulation of the Case 2 retry queue. Returns theory and simulated attempt rates
  (with a batch-means 95 % CI), mean delay, `peak_inflight` and `collapse_triggered`.

---

## Delay solvers

Expected end-to-end sojourn time (including retries) from fixed-point traffic conservation.
They return `float("inf")` (single node / tandem) or `None` entries (feedforward / feedback)
when the network is unstable.

| Function | Returns |
| :--- | :--- |
| `solve_one_node_destruction(lambda_n, p, mu=10.0, T=2.0, backoff=0.03, max_iter=500, tol=1e-11)` | `float` |
| `solve_one_node_modification(lambda_n, p, mu=10.0, T=2.0, max_iter=200, tol=1e-11)` | `float` |
| `solve_tandem_theory(p, N=3, mu=2.0, lambda_arrival=0.15, W=8.0, max_iter=500, tol=1e-10)` | `float` |
| `solve_feedforward_theory(N=3, mu=2.0, lambda_arr=0.15, p=0.1, W=8.0, max_iterations=100, tolerance=1e-6, damping_factor=0.5)` | `(delay, Lambda_0, details)` |
| `solve_feedback_theory(N=5, mu=1.0, lambda_arr=0.05, p=0.1, W=50.0, max_k=200, max_iterations=500, damping=0.5, tolerance=1e-7)` | `(delay, Lambda, details)` |

## Discrete-event simulators

SimPy counterparts of the solvers; each returns the mean simulated sojourn time and accepts
`sim_duration`, `warmup_period` and `seed`.

`simulate_one_node_destruction`, `simulate_one_node_modification`, `simulate_tandem`,
`simulate_feedforward`, `simulate_feedback` — same parameter names as the matching solver.

## Edge-cloud confidence cascade

`CascadeParams`, `solve_cascade_delay`, `solve_cascade_power`, `find_optimal_threshold_tau`,
`solve_batch_timeout_moments`, `jensen_antifragility_bound`, `solve_hysteresis_band`,
`solve_nonstationary_envelope`, `solve_threshold_lambda_sensitivity`,
`solve_pareto_derivative`, `simulate_cascade`, `simulate_stationary_cascade_simpy`,
`simulate_nonstationary_hysteresis_simpy`, `verify_symbolic_cascade_proofs` and helpers.
See the docstrings in `aqnet.models.cascade` (unchanged since 0.2.0).

---

## Command-line interface

```bash
aqnet --version
aqnet bench
aqnet run --topology tandem --nodes 3 --p 0.1 --lambda-arr 0.15 --reps 20
aqnet bifurcation --topology one_node_modification --lambda-arr 3.5 --p 0.25 --mu 10.0 --W 1.0 --json
```
