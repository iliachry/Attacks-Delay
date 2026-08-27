import numpy as np
import simpy
import random
from concurrent.futures import ProcessPoolExecutor

mu = 10.0
T = 2.0
BACKOFF = 0.05
normal_traffic_rates = np.linspace(1, 5, 10)
attack_effectiveness_values = [0.2, 0.5, 0.8]
replications = 60
sim_duration = 20000
warmup_period = 2000

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
            # Backoff happens outside server before re-queuing
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

def run_single(args):
    ln, p, seed = args
    random.seed(seed)
    np.random.seed(seed)
    delays = []
    env = simpy.Environment()
    q = simpy.Store(env)
    srv = simpy.Resource(env, capacity=1)
    
    env.process(packet_generator(env, q, ln))
    env.process(server_process(env, srv, q, mu, p, T, delays))
    
    env.run(until=warmup_period)
    delays.clear()
    env.run(until=warmup_period + sim_duration)
    return np.mean(delays) if delays else np.inf

def sim_destruction(ln, p):
    args = [(ln, p, s) for s in range(replications)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run_single, args))
    valid = [r for r in res if r != np.inf]
    return np.mean(valid) if valid else np.inf

def calculate_theoretic_delay_destruction(lambda_n, p):
    Lambda_star = lambda_n
    for _ in range(500):
        # Effective traffic reaching the server
        lambda_eff = Lambda_star * (1 - p)
        rho = lambda_eff / mu
        if rho >= 1.0:
            return np.inf
        
        # M/G/1 waiting time with zero service for destroyed packets
        E_W = rho / (mu * (1 - rho))
        
        # In M/M/1 queue with effective rate lambda_eff, sojourn time CDF:
        gamma = mu - lambda_eff
        P_complete_in_time = 1 - np.exp(-gamma * T)
        
        # Probability of success in this attempt
        P_succ = (1 - p) * P_complete_in_time
        if P_succ <= 0:
            return np.inf
            
        E_A = 1 / P_succ
        Lambda_star_new = lambda_n * E_A
        if np.isclose(Lambda_star_new, Lambda_star, atol=1e-8):
            break
        Lambda_star = Lambda_star_new
        
    if Lambda_star * (1 - p) >= mu:
        return np.inf
        
    E_A = 1 / P_succ
    # Total delay: (E[A]-1)*(E[W] + BACKOFF) + E[W] + 1/mu
    return E_A * E_W + (E_A - 1) * BACKOFF + (1 / mu)

if __name__ == '__main__':
    print('Testing Corrected Destruction Model:')
    for p in attack_effectiveness_values:
        for ln in [1.0, 2.333333333333333, 3.6666666666666665, 5.0]:
            th = calculate_theoretic_delay_destruction(ln, p)
            sim_v = sim_destruction(ln, p)
            err = abs(th - sim_v) / th * 100
            print(f'p={p:.2f}, ln={ln:.2f} | Theory: {th:.5f}, Sim: {sim_v:.5f}, Rel Err: {err:.2f}%')
