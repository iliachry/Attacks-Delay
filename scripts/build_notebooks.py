"""
Generate the tutorial notebooks in ``notebooks/`` from source cells.

Usage (writes unexecuted notebooks, then executes them in place):

    uv run --extra notebooks python scripts/build_notebooks.py
    uv run --extra notebooks jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent.parent / "notebooks"

SETUP = """import numpy as np
import matplotlib.pyplot as plt

import aqnet

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
print("aqnet", aqnet.__version__)"""


def nb(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    out = new_notebook()
    out.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in cells]
    return out


BASIC = [
    ("md", """# 01 · Basic adversarial queueing networks

`aqnet` models queues where an attacker destroys or corrupts packets with probability `p`
and the sender retransmits after a timeout `T` (called `W` in the multi-node solvers).
Every topology has two tools:

* an **analytical solver** (`solve_*`) based on fixed-point traffic conservation, and
* a **SimPy simulator** (`simulate_*`) of the same system.

This notebook runs both for all five topologies and compares them."""),
    ("code", SETUP),
    ("md", """## All five topologies, theory vs simulation

Each row uses the solver's default operating point. The simulated value is the mean of
four seeds, each run for 20 000 time units, so the numbers are reproducible."""),
    ("code", """SEEDS = range(4)
DUR = 20_000.0


def sim_mean(fn, **kw):
    vals = [fn(sim_duration=DUR, seed=s, **kw) for s in SEEDS]
    return float(np.mean(vals)), float(np.ptp(vals) / 2)


rows = []
kw = dict(lambda_n=2.0, p=0.2, mu=10.0, T=2.0)
rows.append(("1 one-node destruction", aqnet.solve_one_node_destruction(**kw),
             *sim_mean(aqnet.simulate_one_node_destruction, **kw)))
rows.append(("2 one-node modification", aqnet.solve_one_node_modification(**kw),
             *sim_mean(aqnet.simulate_one_node_modification, **kw)))

kw = dict(p=0.1, N=3, mu=2.0, lambda_arrival=0.15, W=8.0)
rows.append(("3 tandem (N=3)", aqnet.solve_tandem_theory(**kw), *sim_mean(aqnet.simulate_tandem, **kw)))

kw = dict(N=3, mu=2.0, lambda_arr=0.15, p=0.1, W=8.0)
rows.append(("4 feedforward (N=3)", aqnet.solve_feedforward_theory(**kw)[0],
             *sim_mean(aqnet.simulate_feedforward, **kw)))

kw = dict(N=5, mu=1.0, lambda_arr=0.05, p=0.1, W=50.0)
rows.append(("5 feedback mesh (N=5)", aqnet.solve_feedback_theory(**kw)[0],
             *sim_mean(aqnet.simulate_feedback, **kw)))

print(f"{'topology':<26}{'theory':>10}{'simulated':>18}{'rel. diff':>11}")
for name, th, sim, half in rows:
    print(f"{name:<26}{th:>10.4f}{sim:>11.4f} ± {half:.4f}{(sim - th) / th:>+11.2%}")"""),
    ("md", """Theory and simulation agree to within a few percent; `±` is half the spread across seeds."""),
    ("md", """## Delay versus load for the single-node cases

*Destruction* (Case 1) kills the packet before service, so the server only sees surviving
traffic. *Modification* (Case 2) corrupts the packet after it has been served, so every
failed attempt still costs a full service time. That difference shows up as much faster
delay growth for Case 2. Solvers return `inf` once no stable operating point exists."""),
    ("code", """p, mu, T = 0.2, 10.0, 2.0
lams = np.linspace(0.2, 9.0, 60)
d_th = [aqnet.solve_one_node_destruction(lam, p, mu=mu, T=T) for lam in lams]
m_th = [aqnet.solve_one_node_modification(lam, p, mu=mu, T=T) for lam in lams]

lam_sim = [1.0, 3.0, 5.0, 6.0]


def runs(fn, lam):
    # A run that collapses into a retry storm records no successes and returns inf.
    vals = np.array([fn(lam, p, mu=mu, T=T, sim_duration=10_000.0, seed=s) for s in range(6)])
    ok = vals[np.isfinite(vals)]
    return (float(ok.mean()) if ok.size else np.nan), int((~np.isfinite(vals)).sum())


d_res = [runs(aqnet.simulate_one_node_destruction, lam) for lam in lam_sim]
m_res = [runs(aqnet.simulate_one_node_modification, lam) for lam in lam_sim]
d_sim = [r[0] for r in d_res]
m_sim = [r[0] for r in m_res]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(lams, d_th, label="Case 1 destruction (theory)")
ax.plot(lams, m_th, label="Case 2 modification (theory)")
ax.scatter(lam_sim, d_sim, marker="o", zorder=3, label="Case 1 SimPy")
ax.scatter(lam_sim, m_sim, marker="s", zorder=3, label="Case 2 SimPy")
ax.set(xlabel="fresh arrival rate λ", ylabel="mean sojourn time", yscale="log",
       title=f"Single node, p={p}, μ={mu}, T={T}")
ax.legend()
plt.show()

for name, topo, curve in (("Case 1", "one_node_destruction", d_th), ("Case 2", "one_node_modification", m_th)):
    crit = aqnet.solve_saddle_node_bifurcation(topo, p=p, mu=mu, T=T).lambda_crit_sn
    last = lams[np.isfinite(curve)][-1]
    print(f"{name}: λ_crit = {crit:.4f}, last finite grid point λ = {last:.4f}")
print(f"{'λ':>4}{'Case 1 sim vs theory':>23}{'Case 2 sim vs theory':>23}{'Case 2 runs collapsed':>24}")
for lam, (d, _), (m, m_col) in zip(lam_sim, d_res, m_res):
    d_err = d / aqnet.solve_one_node_destruction(lam, p, mu=mu, T=T) - 1
    m_err = m / aqnet.solve_one_node_modification(lam, p, mu=mu, T=T) - 1
    print(f"{lam:>4}{d_err:>+23.2%}{m_err:>+23.2%}{m_col:>20} / 6")"""),
    ("md", """* The theory curves end at the saddle-node limit `λ_crit` explained in notebook 03.
* Case 2 matches simulation closely. The Case 1 solver sits a few percent above simulation
  at mid load, and that gap does not shrink with longer runs, so it comes from the
  analytical approximation rather than simulation noise.
* Close to `λ_crit`, some long Case 2 runs **collapse** even though `λ < λ_crit`: a random
  burst of slow services pushes the retry load past the unstable fixed point and the queue
  never recovers (the simulator then reports `inf`; the plot averages the surviving runs).
  Notebook 02 explains this metastability."""),
    ("md", """## Longer chains

In a tandem chain every hop is attacked, so the end-to-end success probability falls like
`(1 - p)^N` and delay rises sharply with `N`."""),
    ("code", """Ns = range(1, 8)
fig, ax = plt.subplots(figsize=(6.5, 4))
for p in (0.0, 0.05, 0.1, 0.15):
    ax.plot(list(Ns), [aqnet.solve_tandem_theory(p=p, N=n, mu=2.0, lambda_arrival=0.15, W=8.0) for n in Ns],
            marker="o", label=f"p={p}")
ax.set(xlabel="hops N", ylabel="mean sojourn time", title="Tandem chain, μ=2, λ=0.15, W=8")
ax.legend()
plt.show()"""),
]


STORM = [
    ("md", """# 02 · Attacks, timeouts and the retry storm

A retransmitted packet is extra load. When attacks or timeouts make attempts fail, the
retries raise the load, which raises queueing delay, which causes more timeouts. This
notebook shows when that feedback loop settles and when it runs away.

We use Case 2 (post-service modification: the packet is served, then rejected) with
service rate `μ = 10` and timeout `T = 1`."""),
    ("code", SETUP),
    ("md", """## The conservation curve Γ(Λ)

Let `Λ` be the attempt rate the server sees (fresh packets plus retries). Only a fraction
`P_succ(Λ)` of attempts succeed, so the useful throughput is `Γ(Λ) = Λ · P_succ(Λ)`.
In steady state that must equal the fresh arrival rate `λ`.

`Γ` rises and then falls: past a point, more attempts mean so much queueing that most of
them time out. The peak is the most fresh traffic the system can carry, `λ_crit`."""),
    ("code", """mu, T = 10.0, 1.0
Lam = np.linspace(1e-3, mu * 0.9999, 600)
fig, ax = plt.subplots(figsize=(6.5, 4))
for p in (0.0, 0.25, 0.5):
    g = [aqnet.throughput_curve_gamma(x, "one_node_modification", p, mu, T) for x in Lam]
    sn = aqnet.solve_saddle_node_bifurcation("one_node_modification", p=p, mu=mu, T=T)
    line, = ax.plot(Lam, g, label=f"p={p}: λ_crit={sn.lambda_crit_sn:.2f}")
    ax.plot(sn.Lambda_sn, sn.lambda_crit_sn, "o", color=line.get_color())
ax.set(xlabel="attempt rate Λ", ylabel="throughput Γ(Λ)", title="Case 2, μ=10, T=1")
ax.legend()
plt.show()"""),
    ("md", """## Timeouts cost capacity even without an attacker

Even at `p = 0`, a short timeout makes slow attempts count as failures. The table compares
`λ_crit` with the naive capacity `μ(1 - p)` that ignores timeouts."""),
    ("code", """print(f"{'μT':>6}{'λ_crit/μ':>11}{'fraction of naive capacity':>30}")
for muT in (0.5, 1, 2, 5, 10, 50):
    sn = aqnet.solve_saddle_node_bifurcation("one_node_modification", p=0.0, mu=10.0, T=muT / 10.0)
    print(f"{muT:>6}{sn.lambda_crit_sn / 10.0:>11.3f}{sn.capacity_ratio_vs_naive:>30.1%}")"""),
    ("md", """## Retry amplification near the limit

On the stable branch, each fresh packet costs `Λ/λ` attempts. Far from the limit that is
just `1/(1 − p)`, the attack overhead; near `λ_crit` timeouts add more retries on top, and
the curve turns vertical at the fold (the square-root law of notebook 03). Mean delay
roughly quadruples across the range."""),
    ("code", """p = 0.25
sn = aqnet.solve_saddle_node_bifurcation("one_node_modification", p=p, mu=mu, T=T)
fr = np.linspace(0.05, 0.999, 80)
amp, delay = [], []
for f in fr:
    d = aqnet.solve_dual_fixed_points(f * sn.lambda_crit_sn, "one_node_modification", p=p, mu=mu, T=T)
    amp.append(d.Lambda_stable / d.lambda_ext)
    delay.append(d.expected_total_time_stable)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
a1.plot(fr, amp); a1.set(xlabel="λ / λ_crit", ylabel="attempts per packet Λ/λ")
a2.plot(fr, delay); a2.set(xlabel="λ / λ_crit", ylabel="mean sojourn time", yscale="log")
fig.suptitle(f"Case 2, p={p}, μ={mu}, T={T}  (λ_crit={sn.lambda_crit_sn:.3f})")
plt.tight_layout(); plt.show()"""),
    ("md", """## The retry storm in simulation

Below `λ_crit` there are **two** fixed points: a stable one and an unstable one above it
(the separatrix). A queue sitting at the stable point recovers from small disturbances,
but a burst that pushes the in-flight retry population past the separatrix never comes
back: retries keep feeding more timeouts until the in-flight population hits the cap.

Four SimPy runs at `λ = 3` (well below `λ_crit ≈ 5.2`) and one above the limit:"""),
    ("code", """runs = [
    ("steady λ=3", dict(lambda_ext=3.0)),
    ("λ=3, 3× burst for 3 s", dict(lambda_ext=3.0, burst_factor=3.0, burst_duration=3.0)),
    ("λ=3, 3× burst for 8 s", dict(lambda_ext=3.0, burst_factor=3.0, burst_duration=8.0)),
    ("steady λ=4.5", dict(lambda_ext=4.5)),
    ("steady λ=5.5 (> λ_crit)", dict(lambda_ext=5.5)),
]
print(f"{'scenario':<26}{'collapsed':>10}{'peak in-flight':>16}{'sim Λ':>8}{'theory Λ_stable':>17}")
for name, kw in runs:
    r = aqnet.simulate_metastable_retry_collapse_simpy(p=p, mu=mu, T=T, sim_time=240.0, seed=7, **kw)
    print(f"{name:<26}{str(r['collapse_triggered']):>10}{r['peak_inflight']:>16}"
          f"{r['sim_attempt_rate_mean']:>8.2f}{r['Lambda_stable_theory']:>17.2f}")"""),
    ("md", """Same burst height, different outcome: for this seed the 3-second burst is absorbed and the
8-second one tips the queue over. Whether a given burst crosses the separatrix is random,
so the next cell sweeps burst length over 24 seeds to estimate the collapse probability."""),
    ("code", """durations = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
seeds = range(24)
prob = []
for dur in durations:
    hits = sum(
        aqnet.simulate_metastable_retry_collapse_simpy(
            lambda_ext=3.0, p=p, mu=mu, T=T, burst_factor=3.0, burst_duration=dur,
            sim_time=120.0, seed=s,
        )["collapse_triggered"]
        for s in seeds
    )
    prob.append(hits / len(seeds))

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(durations, prob, marker="o")
ax.set(xlabel="burst duration (3× arrival rate)", ylabel="P(collapse)", ylim=(-0.05, 1.05),
       title=f"λ=3, p={p}, μ={mu}, T={T}, {len(seeds)} seeds per point")
plt.show()
print({d: round(pr, 3) for d, pr in zip(durations, prob)})"""),
    ("md", """Collapse probability climbs from near zero to one as the burst gets longer. Note that even
with no burst a run occasionally collapses: in a finite stochastic queue the lower branch
is only *metastable*, and a large enough random excursion can cross the separatrix on its
own. The closer `λ` is to `λ_crit`, the narrower the gap and the more often that happens."""),
]


SADDLE = [
    ("md", """# 03 · Saddle-node bifurcation, closed form

The peak of the conservation curve `Γ(Λ)` is a **saddle-node (fold) bifurcation**: as the
fresh rate `λ` rises to `λ_crit`, the stable and unstable fixed points move toward each
other and merge. For single-node queues the fold has an exact solution.

For Case 2, setting `Γ'(Λ) = 0` and writing `v = 1 + ΛT` gives `v·eᵛ = e^(μT+1)`, so

$$v = W_0\\!\\left(e^{\\mu T+1}\\right) = \\omega(\\mu T+1),\\qquad
\\Lambda_{SN} = \\frac{v-1}{T},\\qquad
\\lambda_{crit} = (1-p)\\,\\frac{(v-1)^2}{T\\,v}.$$

`W₀` is the principal Lambert-W branch and `ω` the Wright-omega function."""),
    ("code", SETUP),
    ("md", """## Wright-omega and Lambert-W₀ agree

`fold_parameter_v` uses Wright-omega by default because Lambert-W needs `e^(μT+1)`, which
overflows double precision once `μT > 708`."""),
    ("code", """print(f"{'μT':>8}{'ω(μT+1)':>16}{'W0(e^(μT+1))':>16}{'|diff|':>11}{'v+ln v-(μT+1)':>16}")
for muT in (0.1, 1.0, 10.0, 100.0, 700.0, 5000.0):
    v = aqnet.fold_parameter_v(muT, 1.0)
    try:
        w = aqnet.fold_parameter_v(muT, 1.0, method="lambertw")
        diff = f"{abs(v - w):.1e}"
        w = f"{w:.10g}"
    except ValueError:
        w, diff = "overflow", "-"
    print(f"{muT:>8}{v:>16.10g}{w:>16}{diff:>11}{v + np.log(v) - (muT + 1):>16.1e}")"""),
    ("md", """## Closed form against brute-force maximisation"""),
    ("code", """from scipy.optimize import minimize_scalar

print(f"{'case':<24}{'p':>5}{'λ_crit closed':>15}{'λ_crit numeric':>16}{'rel diff':>10}")
for topo in ("one_node_destruction", "one_node_modification"):
    for p in (0.0, 0.3, 0.6):
        sn = aqnet.solve_saddle_node_bifurcation(topo, p=p, mu=10.0, T=1.0)
        res = minimize_scalar(lambda x: -aqnet.throughput_curve_gamma(x, topo, p, 10.0, 1.0),
                              bounds=(0, sn.Lambda_max), method="bounded", options={"xatol": 1e-12})
        print(f"{topo:<24}{p:>5}{sn.lambda_crit_sn:>15.10f}{-res.fun:>16.10f}"
              f"{abs(sn.lambda_crit_sn + res.fun) / sn.lambda_crit_sn:>10.1e}")"""),
    ("md", """## Phase portrait: two fixed points

At `λ = 3.5` the horizontal line meets `Γ` twice. Between the two crossings `Γ(Λ) > λ`:
more packets leave than arrive, so the retry load shrinks back to the stable point. Above
the separatrix `Γ(Λ) < λ` and the load keeps growing."""),
    ("code", """topo, p, mu, T, lam = "one_node_modification", 0.25, 10.0, 1.0, 3.5
sn = aqnet.solve_saddle_node_bifurcation(topo, p=p, mu=mu, T=T)
d = aqnet.solve_dual_fixed_points(lam, topo, p=p, mu=mu, T=T)
Lam = np.linspace(1e-3, sn.Lambda_max * 0.9999, 600)
g = np.array([aqnet.throughput_curve_gamma(x, topo, p, mu, T) for x in Lam])

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(Lam, g, label="Γ(Λ)")
ax.axhline(lam, color="gray", ls=":", label=f"λ = {lam}")
ax.fill_between(Lam, lam, g, where=(g > lam), alpha=0.15, label="drains back (Γ > λ)")
ax.plot(d.Lambda_stable, lam, "o", color="tab:green", ms=8, label=f"stable Λ={d.Lambda_stable:.3f}")
ax.plot(d.Lambda_unstable, lam, "o", mfc="white", color="tab:red", ms=8,
        label=f"separatrix Λ={d.Lambda_unstable:.3f}")
ax.plot(sn.Lambda_sn, sn.lambda_crit_sn, "k^", label=f"fold ({sn.Lambda_sn:.3f}, {sn.lambda_crit_sn:.3f})")
ax.set(xlabel="attempt rate Λ", ylabel="Γ(Λ)", title=f"Case 2, p={p}, μ={mu}, T={T}")
ax.legend(fontsize=8)
plt.show()"""),
    ("md", """## Bifurcation diagram and the square-root normal form

Close to the fold, `Λ± ≈ Λ_SN ± √((λ_crit − λ)/c₂)` with `c₂ = −Γ''(Λ_SN)/2`. The dashed
curves are that approximation."""),
    ("code", """lams = sn.lambda_crit_sn * (1 - np.geomspace(1, 1e-6, 200))[1:]
duals = [aqnet.solve_dual_fixed_points(l, topo, p=p, mu=mu, T=T) for l in lams]
lo = [x.Lambda_stable for x in duals]; hi = [x.Lambda_unstable for x in duals]
alo = [x.sqrt_asymptotic_stable for x in duals]; ahi = [x.sqrt_asymptotic_unstable for x in duals]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(lams, lo, color="tab:green", label="stable branch")
ax.plot(lams, hi, color="tab:red", label="unstable branch (separatrix)")
ax.plot(lams, alo, "--", color="tab:green", alpha=0.6, label="√ normal form")
ax.plot(lams, ahi, "--", color="tab:red", alpha=0.6)
ax.plot(sn.lambda_crit_sn, sn.Lambda_sn, "k^", label="fold")
ax.axvspan(sn.lambda_crit_sn, sn.lambda_crit_sn * 1.06, color="gray", alpha=0.15, label="no fixed point")
ax.set(xlabel="fresh arrival rate λ", ylabel="attempt rate Λ", ylim=(0, sn.Lambda_max),
       title="Saddle-node bifurcation, Case 2")
ax.legend(fontsize=8)
plt.show()"""),
    ("md", """On log-log axes the branch gap scales with slope `+1/2` and the stable-branch
sensitivity `dΛ/dλ` with slope `−1/2`, the signature of a fold."""),
    ("code", """margins = np.geomspace(1e-1, 1e-6, 30) * sn.lambda_crit_sn
ds = [aqnet.solve_dual_fixed_points(sn.lambda_crit_sn - m, topo, p=p, mu=mu, T=T) for m in margins]
gap = np.array([x.separatrix_gap for x in ds])
sens = np.array([x.delay_sensitivity_dLambda_dlambda for x in ds])
slope_gap = np.polyfit(np.log(margins[10:]), np.log(gap[10:]), 1)[0]
slope_sens = np.polyfit(np.log(margins[10:]), np.log(sens[10:]), 1)[0]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(margins, gap, marker=".", label=f"separatrix gap, slope {slope_gap:+.3f}")
ax.loglog(margins, sens, marker=".", label=f"dΛ/dλ, slope {slope_sens:+.3f}")
ax.set(xlabel="distance to fold  λ_crit − λ", title="Square-root scaling at the fold")
ax.legend()
plt.show()
print(f"fitted slopes: gap {slope_gap:.4f} (theory 0.5), sensitivity {slope_sens:.4f} (theory -0.5)")"""),
    ("md", """## How attacks move the fold

* **Destruction** (Case 1): the attacker removes packets before the server, so the server
  load is `(1 − p)Λ` and `λ_crit` does not depend on `p` at all.
* **Modification** (Case 2): wasted service shrinks `λ_crit` linearly, `(1 − p)·λ_crit(0)`.
* **Multi-node** cases (numerical fold) decay faster because each hop is attacked."""),
    ("code", """ps = np.linspace(0, 0.9, 31)
series = {
    "Case 1 destruction": ("one_node_destruction", 2),
    "Case 2 modification": ("one_node_modification", 2),
    "Case 3 tandem N=3": ("tandem", 3),
    "Case 4 feedforward N=3": ("feedforward", 3),
    "Case 5 feedback N=3": ("feedback", 3),
}
fig, ax = plt.subplots(figsize=(6.5, 4))
for label, (topo_i, N) in series.items():
    ax.plot(ps, [aqnet.solve_saddle_node_bifurcation(topo_i, p=pp, mu=10.0, T=1.0, N=N).lambda_crit_sn
                 for pp in ps], label=label)
ax.set(xlabel="attack probability p", ylabel="λ_crit", title="Critical capacity vs attack, μ=10, T=1")
ax.legend(fontsize=8)
plt.show()"""),
    ("md", """## Symbolic verification

SymPy re-derives the stationarity identity, the curvature at the fold, Case 1
`p`-invariance and the square-root normal form."""),
    ("code", """proofs = aqnet.verify_symbolic_bifurcation_proofs()
for k, val in proofs.items():
    print(f"{k:<36} {val}")
assert proofs["all_symbolic_proofs_verified"]"""),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in (
        ("01_basic_network.ipynb", BASIC),
        ("02_attack_timeout_retry_storm.ipynb", STORM),
        ("03_saddle_node_bifurcation.ipynb", SADDLE),
    ):
        nbformat.write(nb(cells), OUT / name)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
