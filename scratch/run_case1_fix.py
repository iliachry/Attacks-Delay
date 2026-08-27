import numpy as np
import simpy
import random
from concurrent.futures import ProcessPoolExecutor
import json
import os
import matplotlib.pyplot as plt

# --- MODEL PARAMETERS (MATCHING PAPER) ---
mu = 10.0
T = 2.0
BACKOFF = 0.03  # Deterministic backoff B
normal_traffic_rates = np.linspace(1, 5, 10)
attack_effectiveness_values = [0.2, 0.5, 0.8]

# --- SIMULATION PARAMETERS ---
replications = 200
BASE_WARMUP = 2000
sim_duration = 20000

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
        
        # Pre-service destruction check at server
        destroyed = False
        with server.request() as req:
            yield req
            if random.random() < p:
                destroyed = True
            else:
                yield env.timeout(random.expovariate(mu))
        
        if destroyed:
            # Pre-service destruction: server released, packet undergoes backoff B outside server
            if BACKOFF > 0:
                yield env.timeout(BACKOFF)
            packet.attempt_start_time = env.now
            yield queue.put(packet)
        else:
            # Check total attempt timeout (W_q + S)
            if (env.now - packet.attempt_start_time) > T:
                if BACKOFF > 0:
                    yield env.timeout(BACKOFF)
                packet.attempt_start_time = env.now
                yield queue.put(packet)
            else:
                # Successful delivery
                delays.append(env.now - packet.original_arrival_time)

def run_true_simulation(lambda_n, a, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    delays = []
    env = simpy.Environment()
    packet_queue = simpy.Store(env)
    server = simpy.Resource(env, capacity=1)
    
    lambda_eff = lambda_n * (1 - a)
    rho = lambda_eff / mu
    warmup = int(BASE_WARMUP / max(1 - rho, 0.05))
    
    env.process(packet_generator(env, packet_queue, lambda_n))
    env.process(server_process(env, server, packet_queue, mu, a, T, delays))
    
    env.run(until=warmup)
    delays.clear()
    env.run(until=warmup + sim_duration)
    
    return np.mean(delays) if delays else np.inf

def run_multiple_simulations(lambda_n, a, replications):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_true_simulation, lambda_n, a, i)
            for i in range(replications)
        ]
        results = [f.result() for f in futures if f.result() != np.inf]
    return np.mean(results) if results else np.inf

def calculate_theoretic_delay_final(lambda_n, p):
    """
    Theoretical delay calculation for Case 1: Pre-service destruction attacks.
    Matches Section 3.1 of paper.tex exactly.
    """
    Lambda_star = lambda_n
    for _ in range(500):
        lambda_eff = Lambda_star * (1 - p)
        if lambda_eff >= mu:
            return np.inf
            
        rho = lambda_eff / mu
        if rho >= 1.0:
            return np.inf
            
        # M/M/1 waiting time in queue
        E_W = rho / (mu * (1 - rho))
        E_S = 1.0 / mu
        
        # Total sojourn time CDF for M/M/1 attempt: P(W_q + S <= T)
        gamma = mu - lambda_eff
        P_complete_in_time = 1.0 - np.exp(-gamma * T)
        
        P_succ = (1.0 - p) * P_complete_in_time
        if P_succ <= 0:
            return np.inf
            
        E_A = 1.0 / P_succ
        Lambda_star_new = lambda_n * E_A
        
        if np.isclose(Lambda_star_new, Lambda_star, atol=1e-8):
            break
        Lambda_star = Lambda_star_new
        
    if Lambda_star * (1 - p) >= mu:
        return np.inf
        
    E_A = 1.0 / P_succ
    E_D_attempt = E_W + (1.0 - p) * E_S
    # Total delay: E[A] * E[D_attempt] + (E[A] - 1) * BACKOFF
    return E_A * E_D_attempt + (E_A - 1.0) * BACKOFF

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
            print(f"  Simulating lambda_n={ln:.2f}...")
            theory_val = theoretic_results[a][i]

            if theory_val == np.inf:
                simulated_delays.append(np.inf)
            else:
                avg_delay = run_multiple_simulations(ln, a, replications)
                simulated_delays.append(avg_delay)

            sim_val = simulated_delays[-1]
            gap = (
                abs(theory_val - sim_val)
                if theory_val != np.inf and sim_val != np.inf
                else 0
            )
            rel_err = (gap / theory_val * 100) if theory_val not in (0, np.inf) else 0
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
    results_path = os.path.join(script_dir, '1_one_node_destruction', 'results_destruction.json')
    with open(results_path, 'w') as f:
        json.dump(all_data, f, indent=4)

    # Print relative errors per a
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

    plot_path = os.path.join(script_dir, '1_one_node_destruction', f"destroy_no_service_plot_reps{replications}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved to {plot_path}")
