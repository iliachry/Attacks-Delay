"""
Command-line interface for the aqnet package.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from aqnet import __version__
from aqnet.models.bifurcation import (
    solve_dual_fixed_points,
    solve_saddle_node_bifurcation,
    verify_symbolic_bifurcation_proofs,
)
from aqnet.models.cascade import CascadeParams, solve_cascade_delay
from aqnet.models.feedback import solve_feedback_theory
from aqnet.models.feedforward import solve_feedforward_theory
from aqnet.models.one_node import (
    solve_one_node_destruction,
    solve_one_node_modification,
)
from aqnet.models.tandem import solve_tandem_theory
from aqnet.simulation.engine import (
    simulate_cascade,
    simulate_feedback,
    simulate_feedforward,
    simulate_one_node_destruction,
    simulate_one_node_modification,
    simulate_tandem,
)

RETRY_TOPOLOGIES = [
    "one_node_destruction",
    "one_node_modification",
    "tandem",
    "feedforward",
    "feedback",
]


def _json_safe(val: Any) -> Any:
    """Convert non-finite floats (NaN, +/-inf) to None for strict RFC 8259 JSON compliance."""
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        return fval if np.isfinite(fval) else None
    if isinstance(val, dict):
        return {k: _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(v) for v in val]
    return val


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aqnet",
        description="aqnet: Analytical modeling & SimPy simulation of adversarial queueing networks.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run analytical model and simulation")
    run_parser.add_argument(
        "--topology",
        choices=[*RETRY_TOPOLOGIES, "cascade"],
        default="one_node_destruction",
        help="Network topology to analyze/simulate",
    )
    run_parser.add_argument("--nodes", type=int, default=3, help="Number of nodes N")
    run_parser.add_argument("--mu", type=float, default=10.0, help="Service rate mu")
    run_parser.add_argument("--lambda-arr", type=float, default=2.0, help="Arrival rate lambda")
    run_parser.add_argument("--p", type=float, default=0.2, help="Attack probability p")
    run_parser.add_argument("--W", type=float, default=2.0, help="Timeout threshold")
    run_parser.add_argument("--tau", type=float, default=0.55, help="Cascade confidence threshold tau")
    run_parser.add_argument("--batch-size", type=int, default=4, help="Cascade batch size b")
    run_parser.add_argument("--batch-timeout-ms", type=float, default=25.0, help="Cascade batch timeout T_ms")
    run_parser.add_argument("--reps", type=int, default=10, help="Simulation replications")
    run_parser.add_argument("--sim-duration", type=float, default=3000.0, help="Simulation duration")

    # Command: bifurcation
    bif_parser = subparsers.add_parser(
        "bifurcation",
        help="Analyze Saddle-Node (Fold) retry bifurcation, dual fixed points, and SymPy proofs",
    )
    bif_parser.add_argument(
        "--topology",
        choices=RETRY_TOPOLOGIES,
        default="one_node_modification",
        help="Retry queueing topology to analyze",
    )
    bif_parser.add_argument("--nodes", type=int, default=2, help="Number of nodes/branches N")
    bif_parser.add_argument("--mu", type=float, default=10.0, help="Service rate mu")
    bif_parser.add_argument("--lambda-arr", type=float, default=1.5, help="External arrival rate lambda")
    bif_parser.add_argument("--p", type=float, default=0.25, help="Attack probability p")
    bif_parser.add_argument("--W", type=float, default=1.0, help="Timeout threshold T")
    bif_parser.add_argument("--json", action="store_true", help="Emit structured JSON output")

    # Command: bench
    subparsers.add_parser("bench", help="Run fast verification benchmark across all topologies")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "bench":
        print(f"=== aqnet v{__version__} Multi-Topology Verification Benchmark ===")
        th1 = solve_one_node_destruction(lambda_n=2.0, p=0.2, mu=10.0, T=2.0)
        sim1 = np.mean([simulate_one_node_destruction(2.0, 0.2, mu=10.0, T=2.0, seed=i) for i in range(5)])
        print(f"[Case 1: 1-Node Destruction]   Theory: {th1:.4f}s | Sim: {sim1:.4f}s | Err: {abs(th1-sim1)/th1*100:.2f}%")

        th2 = solve_one_node_modification(lambda_n=2.0, p=0.2, mu=10.0, T=2.0)
        sim2 = np.mean([simulate_one_node_modification(2.0, 0.2, mu=10.0, T=2.0, seed=i) for i in range(5)])
        print(f"[Case 2: 1-Node Modification]  Theory: {th2:.4f}s | Sim: {sim2:.4f}s | Err: {abs(th2-sim2)/th2*100:.2f}%")

        th3 = solve_tandem_theory(p=0.05, N=3, mu=2.0, lambda_arrival=0.15, W=8.0)
        sim3 = np.mean([simulate_tandem(0.05, N=3, mu=2.0, lambda_arrival=0.15, W=8.0, seed=i) for i in range(5)])
        print(f"[Case 3: Tandem Chain (N=3)]   Theory: {th3:.4f}s | Sim: {sim3:.4f}s | Err: {abs(th3-sim3)/th3*100:.2f}%")

        th4, _, _ = solve_feedforward_theory(N=3, mu=2.0, lambda_arr=0.15, p=0.05, W=8.0)
        print(f"[Case 4: Feedforward (N=3)]    Theory: {th4:.4f}s")

        th5, _, _ = solve_feedback_theory(N=3, mu=1.0, lambda_arr=0.05, p=0.05, W=50.0)
        sim5 = np.mean([simulate_feedback(N=3, mu=1.0, lambda_arr=0.05, p=0.05, W=50.0, seed=i) for i in range(3)])
        print(f"[Case 5: Feedback Mesh (N=3)]  Theory: {th5:.4f}s | Sim: {sim5:.4f}s | Err: {abs(th5-sim5)/th5*100:.2f}%")

        sn_mod = solve_saddle_node_bifurcation("one_node_modification", p=0.25, mu=10.0, T=1.0)
        print(
            f"[Saddle-Node Fold (Case 2)]    Lambda_SN: {sn_mod.Lambda_sn:.4f} | "
            f"lambda_crit^SN: {sn_mod.lambda_crit_sn:.4f} | c2: {sn_mod.c2_normal_form:.4f}"
        )
        print("Verification complete.")
        return

    if args.command == "bifurcation":
        sn = solve_saddle_node_bifurcation(
            topology=args.topology, p=args.p, mu=args.mu, T=args.W, N=args.nodes
        )
        dual = solve_dual_fixed_points(
            lambda_ext=args.lambda_arr,
            topology=args.topology,
            p=args.p,
            mu=args.mu,
            T=args.W,
            N=args.nodes,
        )
        proofs = verify_symbolic_bifurcation_proofs()
        payload = _json_safe(
            {
                "topology": sn.topology,
                "p": sn.p,
                "mu": sn.mu,
                "T": sn.T,
                "N": sn.N,
                "lambda_ext": args.lambda_arr,
                "Lambda_sn": sn.Lambda_sn,
                "lambda_crit_sn": sn.lambda_crit_sn,
                "rho_sn": sn.rho_sn,
                "c2_normal_form": sn.c2_normal_form,
                "p_crit_for_lambda": sn.p_crit_for_lambda(args.lambda_arr),
                "capacity_ratio_vs_naive": sn.capacity_ratio_vs_naive,
                "is_stable": dual.is_stable,
                "is_critical_fold": dual.is_critical_fold,
                "stability_regime": dual.stability_regime,
                "Lambda_stable": dual.Lambda_stable,
                "Lambda_unstable": dual.Lambda_unstable,
                "separatrix_gap": dual.separatrix_gap,
                "expected_total_time_stable": dual.expected_total_time_stable,
                "symbolic_proofs_verified": proofs["all_symbolic_proofs_verified"],
            }
        )
        if args.json:
            print(json.dumps(payload, indent=2, allow_nan=False))
            return

        print(f"=== aqnet v{__version__} Saddle-Node Retry Bifurcation Analysis ({sn.topology}) ===")
        print(f"Fold Attempt Rate Lambda_SN:     {sn.Lambda_sn:.6f} pkt/s (rho_SN = {sn.rho_sn:.4f})")
        print(f"Critical Arrival lambda_crit^SN: {sn.lambda_crit_sn:.6f} pkt/s ({sn.capacity_ratio_vs_naive*100:.2f}% of naive)")
        print(f"Normal-Form Curvature c_2:       {sn.c2_normal_form:.6f} s")
        print(f"Critical Attack Prob p_crit:     {sn.p_crit_for_lambda(args.lambda_arr):.6f}")
        if dual.is_stable or dual.is_critical_fold:
            print(f"Stable Branch Lambda_stable^*:   {dual.Lambda_stable:.6f} pkt/s")
            print(f"Unstable Separatrix Lambda_u^*:  {dual.Lambda_unstable:.6f} pkt/s (gap = {dual.separatrix_gap:.6f})")
            print(f"Expected Sojourn E[T_total]:     {dual.expected_total_time_stable:.6f} s")
            print(f"Regime Status:                   {dual.stability_regime}")
        else:
            print("Regime Status:                   SUPERCRITICAL COLLAPSE (no stationary fixed point)")
        print(f"SymPy Symbolic Proofs Verified:  {proofs['all_symbolic_proofs_verified']}")
        return

    if args.command == "run":
        print(f"Running aqnet for topology '{args.topology}'...")
        if args.topology == "one_node_destruction":
            theory = solve_one_node_destruction(args.lambda_arr, args.p, mu=args.mu, T=args.W)
            sim_runs = [
                simulate_one_node_destruction(
                    args.lambda_arr, args.p, mu=args.mu, T=args.W,
                    sim_duration=args.sim_duration, seed=i
                )
                for i in range(args.reps)
            ]
        elif args.topology == "one_node_modification":
            theory = solve_one_node_modification(args.lambda_arr, args.p, mu=args.mu, T=args.W)
            sim_runs = [
                simulate_one_node_modification(
                    args.lambda_arr, args.p, mu=args.mu, T=args.W,
                    sim_duration=args.sim_duration, seed=i
                )
                for i in range(args.reps)
            ]
        elif args.topology == "tandem":
            theory = solve_tandem_theory(args.p, N=args.nodes, mu=args.mu, lambda_arrival=args.lambda_arr, W=args.W)
            sim_runs = [
                simulate_tandem(
                    args.p, N=args.nodes, mu=args.mu, lambda_arrival=args.lambda_arr, W=args.W,
                    sim_duration=args.sim_duration, seed=i
                )
                for i in range(args.reps)
            ]
        elif args.topology == "feedback":
            th_val, _, _ = solve_feedback_theory(N=args.nodes, mu=args.mu, lambda_arr=args.lambda_arr, p=args.p, W=args.W)
            theory = float(th_val) if th_val is not None else float("inf")
            sim_runs = [
                simulate_feedback(
                    N=args.nodes, mu=args.mu, lambda_arr=args.lambda_arr, p=args.p, W=args.W,
                    sim_duration=args.sim_duration, seed=i
                )
                for i in range(args.reps)
            ]
        elif args.topology == "cascade":
            params = CascadeParams(lam=args.lambda_arr)
            theory = (
                float(
                    solve_cascade_delay(
                        args.tau, b=args.batch_size, T_ms=args.batch_timeout_ms, params=params
                    )["E_D_ms"]
                )
                / 1000.0
            )
            sim_runs = [
                float(
                    simulate_cascade(
                        tau=args.tau,
                        b=args.batch_size,
                        T_ms=args.batch_timeout_ms,
                        params=params,
                        sim_duration_s=args.sim_duration,
                        seed=i,
                    )["E_D_ms"]
                )
                / 1000.0
                for i in range(args.reps)
            ]
        else:
            th_val, _, _ = solve_feedforward_theory(N=args.nodes, mu=args.mu, lambda_arr=args.lambda_arr, p=args.p, W=args.W)
            theory = float(th_val) if th_val is not None else float("inf")
            sim_runs = [
                simulate_feedforward(
                    N=args.nodes, mu=args.mu, lambda_arr=args.lambda_arr, p=args.p, W=args.W,
                    sim_duration=args.sim_duration, seed=i
                )
                for i in range(args.reps)
            ]

        finite_runs = [r for r in sim_runs if np.isfinite(r)]
        if len(finite_runs) == len(sim_runs) and len(finite_runs) > 0:
            sim_mean = float(np.mean(finite_runs))
            sim_std = float(np.std(finite_runs)) / np.sqrt(len(finite_runs))
        else:
            sim_mean = float("inf")
            sim_std = float("nan")

        print(f"Theory (Closed-Form):   {theory:.5f} s" if np.isfinite(theory) else "Theory: UNSTABLE (inf)")
        if np.isfinite(sim_mean):
            print(f"Simulation ({args.reps} reps): {sim_mean:.5f} ± {sim_std:.5f} s")
        else:
            print(f"Simulation ({args.reps} reps): UNSTABLE (inf)")
        if np.isfinite(theory) and np.isfinite(sim_mean) and theory > 0:
            rel_err = abs(theory - sim_mean) / theory * 100
            print(f"Relative Error:         {rel_err:.2f}%")


if __name__ == "__main__":
    main()
