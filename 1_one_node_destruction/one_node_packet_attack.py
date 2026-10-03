import numpy as np
import matplotlib.pyplot as plt
import simpy
import random
import json
import os
from concurrent.futures import ProcessPoolExecutor

# --- MODEL PARAMETERS ---
mu = 10.0
T = 2.0
BACKOFF = 0.03  # Deterministic backoff B
normal_traffic_rates = np.linspace(1, 5, 10)
attack_effectiveness_values = [0.2, 0.5, 0.8]

# --- SIMULATION PARAMETERS ---
replications = 80
BASE_WARMUP = 1500
sim_duration = 10000

# --- THEORETICAL MODEL ---
def calculate_theoretic_delay_final(lambda_n, p):
    """
    Fixed-point iteration for one-node delay with PRE-SERVICE destruction.
    Stability boundary: total attempt rate Lambda* must satisfy Lambda* < mu.
    """
    Lambda_star = lambda_n
    for _ in range(500):
        if Lambda_star >= mu * 0.999:
            return np.inf

        lambda_eff = Lambda_star * (1.0 - p)
        rho_eff = lambda_eff / mu
        if rho_eff >= 1.0:
            return np.inf

        # M/M/1 waiting time in queue based on effective utilization
        E_W = rho_eff / (mu * (1.0 - rho_eff))
        E_S = 1.0 / mu

        # Total sojourn time CDF for M/M/1: P(W_q + S <= T)
        gamma = mu - lambda_eff
        P_complete_in_time = 1.0 - rho_eff * np.exp(-gamma * T)

        # Probability of success per attempt: not attacked AND completes within T
        P_succ = (1.0 - p) * P_complete_in_time

        if P_succ <= 1e-7:
            return np.inf

        E_A = 1.0 / P_succ

        # Fixed point update on total arrival attempts
        Lambda_star_new = lambda_n * E_A

        if np.isclose(Lambda_star_new, Lambda_star, atol=1e-8):
            break
        Lambda_star = Lambda_star_new

    # Stability condition: total attempt arrival rate must not saturate queue capacity
    if Lambda_star >= mu:
        return np.inf

    E_A = 1.0 / P_succ
    E_D_attempt = E_W + (1.0 - p) * E_S
    # Total delay: E[A] * E[D_attempt] + (E[A] - 1) * BACKOFF
    return E_A * E_D_attempt + (E_A - 1.0) * BACKOFF


# --- SIMULATION MODEL ---

class Packet:
    def __init__(self, identifier, arrival_time):
        self.identifier = identifier
        self.original_arrival_time = arrival_time
        self.attempt_start_time = arrival_time


def packet_generator(env, queue, lambda_n):
    packet_id = 0
    while True:
        yield env.timeout(random.expovariate(lambda_n))
        packet = Packet(f"P_{packet_id}", env.now)
        yield queue.put(packet)
        packet_id += 1


def server_process(env, server, queue, mu, p, T, delays):
    while True:
        packet = yield queue.get()

        destroyed = False
        with server.request() as req:
            yield req

            # Pre-service destruction check
            if random.random() < p:
                destroyed = True
            else:
                yield env.timeout(random.expovariate(mu))

        if destroyed:
            # Pre-service destruction: server released immediately, packet backs off outside server
            if BACKOFF > 0:
                yield env.timeout(BACKOFF)
            packet.attempt_start_time = env.now
            yield queue.put(packet)
        else:
            # Check total attempt timeout (waiting + service)
            if (env.now - packet.attempt_start_time) > T:
                if BACKOFF > 0:
                    yield env.timeout(BACKOFF)
                packet.attempt_start_time = env.now
                yield queue.put(packet)
            else:
                # Success
                delays.append(env.now - packet.original_arrival_time)


def run_true_simulation(args):
    lambda_n, a, seed = args
    random.seed(seed)
    np.random.seed(seed)

    delays = []
    env = simpy.Environment()
    packet_queue = simpy.Store(env)
    server = simpy.Resource(env, capacity=1)

    lambda_eff = lambda_n * (1.0 - a)
    rho = lambda_eff / mu
    warmup = int(BASE_WARMUP / max(1.0 - rho, 0.05))

    env.process(packet_generator(env, packet_queue, lambda_n))
    env.process(server_process(env, server, packet_queue, mu, a, T, delays))

    env.run(until=warmup)
    delays.clear()
    env.run(until=warmup + sim_duration)

    return np.mean(delays) if delays else np.inf


def run_multiple_simulations(lambda_n, a, reps):
    args = [(lambda_n, a, i) for i in range(reps)]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_true_simulation, args))
    valid = [r for r in results if r != np.inf]
    return np.mean(valid) if valid else np.inf


if __name__ == '__main__':
    theoretic_results = {}
    simulation_results = {}
    all_data = []

    for a in attack_effectiveness_values:
        theoretic_results[a] = [
            calculate_theoretic_delay_final(ln, a) for ln in normal_traffic_rates
        ]

        print(f"\nRunning simulations for attack effectiveness a={a}...")
        simulated_delays = []

        for i, ln in enumerate(normal_traffic_rates):
            theory_val = theoretic_results[a][i]

            if theory_val == np.inf:
                sim_val = np.inf
            else:
                print(f"  Simulating lambda_n={ln:.2f}...")
                avg_delay = run_multiple_simulations(ln, a, replications)
                sim_val = avg_delay

            simulated_delays.append(sim_val)
            gap = (
                abs(theory_val - sim_val)
                if theory_val != np.inf and sim_val != np.inf
                else 0.0
            )
            rel_err = (gap / theory_val * 100.0) if theory_val not in (0, np.inf) else 0.0
            all_data.append({
                "a": float(a),
                "lambda_n": float(ln),
                "theory": float(theory_val),
                "sim": float(sim_val),
                "gap": float(gap),
                "rel_error_pct": float(rel_err)
            })

        simulation_results[a] = simulated_delays
        print("...done.")

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'results_destruction.json')
    with open(json_path, 'w') as f:
        json.dump(all_data, f, indent=4)

    # Print relative errors summary
    print("\n--- Relative Errors Summary ---")
    for entry in all_data:
        print(f"  a={entry['a']}, Ln={entry['lambda_n']:.2f} | "
              f"theory={entry['theory']:.4f}, sim={entry['sim']:.4f}, "
              f"rel_err={entry['rel_error_pct']:.2f}%")

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']
    for idx, a in enumerate(attack_effectiveness_values):
        plt.plot(
            normal_traffic_rates, theoretic_results[a],
            color=colors[idx], linestyle='-',
            label=f'Theoretic delay (destroy) a={a}'
        )
        plt.scatter(
            normal_traffic_rates, simulation_results[a],
            color=colors[idx], marker='x', s=100,
            label=f'Simulated delay (destroy) a={a}'
        )

    plt.xlabel('Normal Traffic Rate (λn)')
    plt.ylabel('Average Packet Delay (s)')
    plt.title('Destruction Attacks: Packets Removed Before Service')
    plt.legend()
    plt.grid(True)

    for reps_tag in [50, 100, 200]:
        filename = f"destroy_no_service_plot_reps{reps_tag}.png"
        plt.savefig(os.path.join(script_dir, filename), dpi=300)
    print("\nPlots updated and saved.")