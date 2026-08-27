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
BACKOFF = 0.03
normal_traffic_rates = np.linspace(1, 5, 10)
attack_effectiveness_values = [0.2, 0.5, 0.8]

# --- SIMULATION PARAMETERS ---
replications = 60
BASE_WARMUP = 1500
sim_duration = 15000

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
            # Check pre-service destruction
            if random.random() < p:
                destroyed = True
            else:
                yield env.timeout(random.expovariate(mu))
        
        if destroyed:
            if BACKOFF > 0:
                yield env.timeout(BACKOFF)
            packet.attempt_start_time = env.now
            yield queue.put(packet)
        else:
            if (env.now - packet.attempt_start_time) > T:
                if BACKOFF > 0:
                    yield env.timeout(BACKOFF)
                packet.attempt_start_time = env.now
                yield queue.put(packet)
            else:
                delays.append(env.now - packet.original_arrival_time)

def run_single_sim(args):
    lambda_n, p, seed = args
    random.seed(seed)
    np.random.seed(seed)
    
    delays = []
    env = simpy.Environment()
    packet_queue = simpy.Store(env)
    server = simpy.Resource(env, capacity=1)
    
    # Calculate utilization to set warmup
    Lambda_star = lambda_n / (1 - p)
    rho = min(Lambda_star / mu, 0.95)
    warmup = int(BASE_WARMUP / max(1 - rho, 0.05))
    
    env.process(packet_generator(env, packet_queue, lambda_n))
    env.process(server_process(env, server, packet_queue, mu, p, T, delays))
    
    env.run(until=warmup)
    delays.clear()
    env.run(until=warmup + sim_duration)
    
    return np.mean(delays) if delays else np.inf

def calculate_theoretic_delay_final(lambda_n, p):
    Lambda_star = lambda_n
    for _ in range(500):
        if Lambda_star >= mu * 0.999:
            return np.inf
            
        rho = Lambda_star / mu
        if rho >= 1.0:
            return np.inf
            
        gamma = mu - Lambda_star
        P_complete_in_time = 1.0 - np.exp(-gamma * T)
        P_succ = (1.0 - p) * P_complete_in_time
        if P_succ <= 0:
            return np.inf
            
        E_A = 1.0 / P_succ
        Lambda_star_new = lambda_n * E_A
        if np.isclose(Lambda_star_new, Lambda_star, atol=1e-8):
            break
        Lambda_star = Lambda_star_new
        
    if Lambda_star >= mu:
        return np.inf
        
    E_W = rho / (mu * (1.0 - rho))
    E_S = 1.0 / mu
    E_D_attempt = E_W + (1.0 - p) * E_S
    return E_A * E_D_attempt + (E_A - 1.0) * BACKOFF

if __name__ == '__main__':
    all_data = []
    theoretic_results = {}
    simulation_results = {}
    
    for a in attack_effectiveness_values:
        print(f"\n--- Testing a={a} ---")
        theoretic_results[a] = []
        simulation_results[a] = []
        
        for ln in normal_traffic_rates:
            th = calculate_theoretic_delay_final(ln, a)
            theoretic_results[a].append(th)
            
            if th == np.inf:
                sim_val = np.inf
            else:
                args = [(ln, a, s) for s in range(replications)]
                with ProcessPoolExecutor() as ex:
                    res = list(ex.map(run_single_sim, args))
                valid = [r for r in res if r != np.inf]
                sim_val = np.mean(valid) if valid else np.inf
                
            simulation_results[a].append(sim_val)
            gap = abs(th - sim_val) if th != np.inf and sim_val != np.inf else 0.0
            rel_err = (gap / th * 100.0) if th not in (0, np.inf) else 0.0
            print(f"a={a}, ln={ln:.2f} | Theory: {th:.4f}, Sim: {sim_val:.4f}, RelErr: {rel_err:.2f}%")
            
            all_data.append({
                "a": float(a),
                "lambda_n": float(ln),
                "theory": float(th),
                "sim": float(sim_val),
                "gap": float(gap),
                "rel_error_pct": float(rel_err)
            })
            
    print("Done testing.")
