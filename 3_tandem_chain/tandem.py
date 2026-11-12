import numpy as np
import matplotlib.pyplot as plt
import simpy
import random
import json

# --- MODEL PARAMETERS ---
mu = 2.0
lambda_arrival = 0.15
W = 8.0
N_tandem = 3
p_values = [0.05, 0.1, 0.15, 0.2]

# --- SIMULATION PARAMETERS ---
replications = 30
warmup_period = 1000
sim_duration = 5000

class Packet:
    def __init__(self, identifier, arrival_time):
        self.identifier = identifier
        self.original_arrival_time = arrival_time
        self.attempt_start_time = arrival_time
        self.is_attacked = False

def packet_generator(env, queue, lambda_n):
    packet_id = 0
    while True:
        yield env.timeout(random.expovariate(lambda_n))
        packet = Packet(f"P_{packet_id}", env.now)
        yield queue.put(packet)
        packet_id += 1

def server_process(env, servers, queues, node_id, p):
    while True:
        packet = yield queues[node_id].get()
        with servers[node_id].request() as req:
            yield req
            
            # Service
            yield env.timeout(random.expovariate(mu))
            
            # Stage corruption (verified end-to-end at destination node N_tandem - 1)
            if random.random() < p:
                packet.is_attacked = True
            
            if node_id < N_tandem - 1:
                yield queues[node_id + 1].put(packet)
            else:
                # End-to-end checksum and timeout verification at destination node
                if packet.is_attacked or (env.now - packet.attempt_start_time) > W:
                    packet.attempt_start_time = env.now
                    packet.is_attacked = False
                    yield queues[0].put(packet)
                else:
                    # Final success
                    packet_delays.append(env.now - packet.original_arrival_time)

def run_tandem_simulation(p):
    global packet_delays
    packet_delays = []
    
    env = simpy.Environment()
    queues = [simpy.Store(env) for _ in range(N_tandem)]
    servers = [simpy.Resource(env, capacity=1) for _ in range(N_tandem)]
    
    env.process(packet_generator(env, queues[0], lambda_arrival))
    for i in range(N_tandem):
        env.process(server_process(env, servers, queues, i, p))
    
    env.run(until=warmup_period)
    packet_delays = []
    env.run(until=warmup_period + sim_duration)
    
    return np.mean(packet_delays) if packet_delays else np.inf

from scipy.stats import gamma

def solve_tandem_theory(p, N=N_tandem, mu=mu, lambda_arrival=lambda_arrival, W=W, max_iter=500, tol=1e-10):
    if p < 0.0 or p >= 1.0 or N < 1 or lambda_arrival < 0.0 or lambda_arrival >= mu or W <= 0.0:
        return float(np.inf)

    Lambda = [lambda_arrival] * N
    P_succ_overall = 1.0

    for _ in range(max_iter):
        if any(Lambda[j] >= mu * 0.999999 for j in range(N)):
            return float(np.inf)

        E_S_total = sum(1.0 / (mu - Lambda[j]) for j in range(N))
        Var_S_total = sum(1.0 / (mu - Lambda[j]) ** 2 for j in range(N))

        if Var_S_total <= 0:
            return float(np.inf)

        k_total = E_S_total**2 / Var_S_total
        theta_total = Var_S_total / E_S_total
        P_succ_overall = ((1.0 - p) ** N) * gamma.cdf(W, a=k_total, scale=theta_total)

        if P_succ_overall <= 1e-7:
            return float(np.inf)

        Lambda_new = [lambda_arrival / P_succ_overall] * N

        if abs(Lambda_new[0] - Lambda[0]) <= tol:
            Lambda = Lambda_new
            break
        Lambda = Lambda_new
    else:
        return float(np.inf)

    if any(Lambda[j] >= mu for j in range(N)) or P_succ_overall <= 1e-7:
        return float(np.inf)

    return float((1.0 / P_succ_overall) * sum(1.0 / (mu - Lambda[j]) for j in range(N)))

# --- MAIN ---
if __name__ == "__main__":
    p_range = np.linspace(0.01, 0.25, 10)
    theory_delays = [solve_tandem_theory(p) for p in p_range]
    sim_delays = []

    print("Running Tandem N=3 Simulation...")
    for p in p_range:
        results = []
        for r in range(replications):
            random.seed(r)
            np.random.seed(r)
            results.append(run_tandem_simulation(p))
        sim_delays.append(np.mean(results))

    plt.figure(figsize=(10, 6))
    plt.plot(p_range, theory_delays, 'b-', label='Theory (Gamma Approx)')
    plt.scatter(p_range, sim_delays, color='red', marker='x', s=100, label='Simulation')
    plt.xlabel('Attack Probability (p)')
    plt.ylabel('Average Delay (s)')
    plt.title(f'Tandem Network (N={N_tandem}) - Delay vs p')
    plt.legend()
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'corrected_tandem_simulation_N3.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Done. Plot saved to {plot_path}")
