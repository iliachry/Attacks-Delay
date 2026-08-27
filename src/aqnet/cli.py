"""
Command-line interface for the aqnet package.
"""

import argparse
import sys
import numpy as np
from aqnet import __version__
from aqnet.models.one_node import solve_one_node_destruction, solve_one_node_modification
from aqnet.models.tandem import solve_tandem_theory
from aqnet.models.feedforward import solve_feedforward_theory
from aqnet.models.feedback import solve_feedback_theory
from aqnet.simulation.engine import (
    simulate_one_node_destruction,
    simulate_one_node_modification,
    simulate_tandem,
    simulate_feedback,
)


def main():
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
        choices=["one_node_destruction", "one_node_modification", "tandem", "feedforward", "feedback"],
        default="one_node_destruction",
        help="Network topology to analyze/simulate",
    )
    run_parser.add_argument("--nodes", type=int, default=3, help="Number of nodes N")
    run_parser.add_argument("--mu", type=float, default=10.0, help="Service rate mu")
    run_parser.add_argument("--lambda-arr", type=float, default=2.0, help="Arrival rate lambda")
    run_parser.add_argument("--p", type=float, default=0.2, help="Attack probability p")
    run_parser.add_argument("--W", type=float, default=2.0, help="Timeout threshold")
    run_parser.add_argument("--reps", type=int, default=10, help="Simulation replications")
    run_parser.add_argument("--sim-duration", type=float, default=3000.0, help="Simulation duration")

    # Command: bench
    subparsers.add_parser("bench", help="Run fast verification benchmark across all topologies")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "bench":
        print(f"=== aqnet v{__version__} Multi-Topology Verification Benchmark ===")
        # Case 1
        th1 = solve_one_node_destruction(lambda_n=2.0, p=0.2, mu=10.0, T=2.0)
        sim1 = np.mean([simulate_one_node_destruction(2.0, 0.2, mu=10.0, T=2.0, seed=i) for i in range(5)])
        print(f"[Case 1: 1-Node Destruction]   Theory: {th1:.4f}s | Sim: {sim1:.4f}s | Err: {abs(th1-sim1)/th1*100:.2f}%")

        # Case 2
        th2 = solve_one_node_modification(lambda_n=2.0, p=0.2, mu=10.0, T=2.0)
        sim2 = np.mean([simulate_one_node_modification(2.0, 0.2, mu=10.0, T=2.0, seed=i) for i in range(5)])
        print(f"[Case 2: 1-Node Modification]  Theory: {th2:.4f}s | Sim: {sim2:.4f}s | Err: {abs(th2-sim2)/th2*100:.2f}%")

        # Case 3
        th3 = solve_tandem_theory(p=0.05, N=3, mu=2.0, lambda_arrival=0.15, W=8.0)
        sim3 = np.mean([simulate_tandem(0.05, N=3, mu=2.0, lambda_arrival=0.15, W=8.0, seed=i) for i in range(5)])
        print(f"[Case 3: Tandem Chain (N=3)]   Theory: {th3:.4f}s | Sim: {sim3:.4f}s | Err: {abs(th3-sim3)/th3*100:.2f}%")

        # Case 4
        th4, _, _ = solve_feedforward_theory(N=3, mu=2.0, lambda_arr=0.15, p=0.05, W=8.0)
        print(f"[Case 4: Feedforward (N=3)]    Theory: {th4:.4f}s")

        # Case 5
        th5, _, _ = solve_feedback_theory(N=3, mu=1.0, lambda_arr=0.05, p=0.05, W=50.0)
        sim5 = np.mean([simulate_feedback(N=3, mu=1.0, lambda_arr=0.05, p=0.05, W=50.0, seed=i) for i in range(3)])
        print(f"[Case 5: Feedback Mesh (N=3)]  Theory: {th5:.4f}s | Sim: {sim5:.4f}s | Err: {abs(th5-sim5)/th5*100:.2f}%")
        print("Verification complete.")
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
            theory, _, _ = solve_feedback_theory(N=args.nodes, mu=args.mu, lambda_arr=args.lambda_arr, p=args.p, W=args.W)
            sim_runs = [
                simulate_feedback(
                    N=args.nodes, mu=args.mu, lambda_arr=args.lambda_arr, p=args.p, W=args.W,
                    sim_duration=args.sim_duration, seed=i
                )
                for i in range(args.reps)
            ]
        else:
            theory, _, _ = solve_feedforward_theory(N=args.nodes, mu=args.mu, lambda_arr=args.lambda_arr, p=args.p, W=args.W)
            sim_runs = [
                simulate_tandem(
                    args.p, N=args.nodes, mu=args.mu, lambda_arrival=args.lambda_arr, W=args.W,
                    sim_duration=args.sim_duration, seed=i
                )
                for i in range(args.reps)
            ]

        sim_mean = float(np.mean(sim_runs))
        sim_std = float(np.std(sim_runs)) / np.sqrt(len(sim_runs))
        print(f"Theory (Closed-Form):   {theory:.5f} s" if np.isfinite(theory) else "Theory: UNSTABLE (inf)")
        print(f"Simulation ({args.reps} reps): {sim_mean:.5f} ± {sim_std:.5f} s")
        if np.isfinite(theory) and theory > 0:
            rel_err = abs(theory - sim_mean) / theory * 100
            print(f"Relative Error:         {rel_err:.2f}%")


if __name__ == "__main__":
    main()
