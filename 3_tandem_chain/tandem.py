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
            
            # Start of attempt at node 0
            if node_id == 0:
                packet.attempt_start_time = env.now
            
            # Service
            yield env.timeout(random.expovariate(mu))
            
            # Check attack or timeout
            if random.random() < p or (env.now - packet.attempt_start_time) > W:
                # Failure: Restart from node 0
                yield queues[0].put(packet)
            else:
                # Success at this node
                if node_id < N_tandem - 1:
                    yield queues[node_id + 1].put(packet)
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

def solve_tandem_theory(p):
    Lambda = [lambda_arrival] * N_tandem
    for _ in range(100):
        # Calculate journey times and timeout probs
        for i in range(N_tandem):
            # Sum of delays up to i
            E_S = sum(1/(mu - Lambda[j]) for j in range(i+1))
            Var_S = sum(1/(mu - Lambda[j])**2 for j in range(i+1))
            
            # Gamma approx
            k = E_S**2 / Var_S
            theta = Var_S / E_S
            P_timeout = 1 - gamma.cdf(W, a=k, scale=theta)
            
            # Rate at node i+1 (if exists) is Lambda_i * (1-p) * (1-P_timeout) ?
            # No, retransmission is from 0. 
            # Total attempts rate Lambda_star = lambda / P_succ
            
        # Overall success probability
        E_S_total = sum(1/(mu - Lambda[j]) for j in range(N_tandem))
        Var_S_total = sum(1/(mu - Lambda[j])**2 for j in range(N_tandem))
        k_total = E_S_total**2 / Var_S_total
        theta_total = Var_S_total / E_S_total
        P_succ_overall = (1-p)**N_tandem * gamma.cdf(W, a=k_total, scale=theta_total)
        
        Lambda_new = [lambda_arrival / P_succ_overall] * N_tandem
        
        if np.isclose(Lambda_new[0], Lambda[0], rtol=1e-5):
            break
        Lambda = Lambda_new
        
    return (1/P_succ_overall) * sum(1/(mu - Lambda[j]) for j in range(N_tandem))

# --- MAIN ---
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
plt.grid(True)
plt.savefig('tandem/tandem_simulation_vs_theory_N3.png')
print("Done. Plot saved in tandem folder.")
