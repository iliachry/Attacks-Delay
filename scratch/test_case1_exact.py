import numpy as np
import simpy
import random
from concurrent.futures import ProcessPoolExecutor
import json
import os
import matplotlib.pyplot as plt

# --- MODEL PARAMETERS ---
mu = 10.0
T = 2.0
normal_traffic_rates = np.linspace(1, 5, 10)
attack_effectiveness_values = [0.2, 0.5, 0.8]

# --- SIMULATION PARAMETERS ---
replications = 100
warmup_period = 2000
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
        
        # In destruction attack, adversary destroys packet with prob p
        # If destroyed, it consumes no service time at the server
        destroyed = False
        with server.request() as req:
            yield req
            if random.random() < p:
                destroyed = True
            else:
                yield env.timeout(random.expovariate(mu))
        
        if destroyed:
            # Failed attempt due to destruction: re-queue for next attempt
            packet.attempt_start_time = env.now
            yield queue.put(packet)
        else:
            # Check timeout on completed service
            if (env.now - packet.attempt_start_time) > T:
                packet.attempt_start_time = env.now
                yield queue.put(packet)
            else:
                # Success
                delays.append(env.now - packet.original_arrival_time)

def run_single_sim(args):
    lambda_n, p, seed = args
    random.seed(seed)
    np.random.seed(seed)
    
    delays = []
    env = simpy.Environment()
    queue = simpy.Store(env)
    server = simpy.Resource(env, capacity=1)
    
    env.process(packet_generator(env, queue, lambda_n))
    env.process(server_process(env, server, queue, mu, p, T, delays))
    
    env.run(until=warmup_period)
    delays.clear()
    env.run(until=warmup_period + sim_duration)
    
    return np.mean(delays) if delays else np.inf

def run_simulations(lambda_n, p):
    args = [(lambda_n, p, s) for s in range(replications)]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_sim, args))
    valid = [r for r in results if r != np.inf]
    return np.mean(valid) if valid else np.inf

def calculate_theoretic_destruction_delay(lambda_n, p):
    """
    Theoretical delay for pre-service destruction attack.
    In destruction attack:
    - Packets arrive with external rate lambda_n.
    - An attempt fails (pre-service destruction) with probability p, requiring retransmission.
    - Effective arrival rate of attempts entering queue: Lambda* = lambda_n / P_succ.
    - Only non-destroyed packets consume service time (1/mu).
    - Rate of packets served by server: lambda_eff = Lambda* * (1 - p) = lambda_n.
    - Server utilization: rho = lambda_eff / mu = lambda_n / mu.
    - For M/M/1 with effective served rate lambda_eff, the sojourn time of a successful attempt is:
      E[D_succ] = 1 / (mu - lambda_eff).
    - Total expected delay under geometric attempts A ~ Geom(P_succ):
      E[D] = E[A] * E[D_attempt] = 1/(1-p) * (1 / (mu - lambda_n))
    """
    if lambda_n >= mu:
        return np.inf
        
    Lambda_star = lambda_n
    for _ in range(500):
        lambda_eff = Lambda_star * (1 - p)
        if lambda_eff >= mu:
            return np.inf
            
        gamma = mu - lambda_eff
        # Sojourn time CDF for M/M/1: P(S <= T) = 1 - exp(-gamma * T)
        P_succ_time = 1 - np.exp(-gamma * T)
        P_succ = (1 - p) * P_succ_time
        if P_succ <= 0:
            return np.inf
            
        E_A = 1 / P_succ
        Lambda_star_new = lambda_n * E_A
        if np.isclose(Lambda_star_new, Lambda_star, atol=1e-8):
            break
        Lambda_star = Lambda_star_new
        
    # Expected delay per attempt: in an M/M/1 system with effective rate lambda_eff = lambda_n,
    # the expected sojourn time is 1 / (mu - lambda_eff).
    # Since retransmitted attempts experience the queue with effective rate lambda_eff:
    E_D = E_A * (1 / (mu - lambda_n))
    # Wait, let's also test if E[D] = (E[A]-1)*E[W] + E[W] + 1/mu or E[A]/(mu - lambda_n)
    return E_D

if __name__ == '__main__':
    print("Testing Case 1 Theory vs Simulation...")
    for p in attack_effectiveness_values:
        for ln in [1.0, 2.0, 3.0, 4.0, 5.0]:
            th = calculate_theoretic_destruction_delay(ln, p)
            sim = run_simulations(ln, p)
            err = abs(th - sim) / th * 100
            print(f"p={p:.2f}, ln={ln:.1f} | Theory: {th:.5f}, Sim: {sim:.5f}, Err: {err:.2f}%")
