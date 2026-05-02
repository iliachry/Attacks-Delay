import numpy as np
import matplotlib.pyplot as plt
import json
import simpy
import random
from scipy.special import gammainc

# --- MODEL PARAMETERS FOR SECTION 3.3.1 ---
mu = 1.0  # Service rate
lambda_arrival = 0.05  # External arrival rate per node
W = 50.0  # Timeout period
N_values = range(2, 11)
p_values = [0.05, 0.1, 0.15, 0.2]

# Simulation parameters
replications = 30
warmup_period = 1000
sim_duration = 5000

def get_erlang_cdf(k, rate, w):
    if k <= 0: return 1.0
    return gammainc(k, rate * w)

def solve_feedback_network_theory(N, mu, lambda_arr, p, W):
    def get_metrics(lambda_star_val):
        gamma_rate = mu - lambda_star_val
        if gamma_rate <= 1e-6:
            return 0, 0, 0, 0
        p_succ = 0
        e_v = 0
        e_d_succ_num = 0
        e_d_fail_num = 0
        max_k = 200
        
        for k in range(1, max_k):
            # term: prob of reaching node k without attack/timeout in 1..k-1 and continuing k-1 times
            # Note: attack Discovery happens AFTER service. 
            # So to reach node k, we must have NOT been attacked in nodes 1..k-1.
            term = ((1 - p) ** (k - 1)) * ((N / (N + 1)) ** (k - 1))
            f_k_minus_1 = get_erlang_cdf(k - 1, gamma_rate, W)
            f_k = get_erlang_cdf(k, gamma_rate, W)
            f_k_plus_1 = get_erlang_cdf(k + 1, gamma_rate, W)
            
            # 1. Expected visits reaching the server (includes the one where it might be attacked/timeout)
            e_v += term * f_k_minus_1
            
            # 2. Probability of success at visit k
            # Must NOT be attacked at k, MUST choose exit, MUST NOT timeout at k
            p_succ_k = term * (1 - p) * (1 / (N + 1)) * f_k
            p_succ += p_succ_k
            e_d_succ_num += p_succ_k * (k / gamma_rate) * (f_k_plus_1 / f_k if f_k > 0 else 1)
            
            # 3. Probability of failure at visit k (Attack or Timeout)
            # Attack at k: prob reach k * attacked at k
            p_attack_k = term * p * f_k_minus_1
            e_d_fail_num += p_attack_k * (k / gamma_rate) # Spent visit k
            
            # Timeout at k: prob reach k * not attacked at k * (not exit or exit) * timeout
            # prob not exit or exit = 1.
            # Timeout happens if S_k > W given S_{k-1} <= W
            p_timeout_k = term * (1 - p) * (f_k_minus_1 - f_k)
            e_d_fail_num += p_timeout_k * (k / gamma_rate)
            
        return p_succ, e_v, e_d_succ_num, e_d_fail_num

    # Fixed-point iteration
    lambda_star_sol = lambda_arr * (N + 1)
    damping = 0.5
    for _ in range(500):
        if lambda_star_sol >= mu * 0.999:
            return None, None, None
        p_succ, e_v, e_d_succ_num, e_d_fail_num = get_metrics(lambda_star_sol)
        if p_succ <= 1e-6:
            return None, None, None
        next_val = lambda_arr * (e_v / p_succ)
        if abs(next_val - lambda_star_sol) < 1e-7:
            lambda_star_sol = next_val
            break
        lambda_star_sol = damping * next_val + (1 - damping) * lambda_star_sol
    else:
        return None, None, None

    p_succ, e_v, e_d_succ_num, e_d_fail_num = get_metrics(lambda_star_sol)
    e_attempts = 1 / p_succ
    e_d_succ = e_d_succ_num / p_succ
    e_d_fail = e_d_fail_num / (1 - p_succ) if p_succ < 1 else 0
    avg_sojourn_time = (e_attempts - 1) * e_d_fail + e_d_succ
    return avg_sojourn_time, lambda_star_sol, (None, None, lambda_star_sol)

class NetworkPacket:
    def __init__(self, packet_id, entry_node, arrival_time):
        self.packet_id = packet_id
        self.entry_node = entry_node
        self.original_arrival_time = arrival_time
        self.attempt_start_time = arrival_time
        self.current_node = entry_node

def run_simulation(N, mu, lambda_arr, p, W):
    def sim_func():
        env = simpy.Environment()
        queues = [simpy.Store(env) for _ in range(N)]
        servers = [simpy.Resource(env, capacity=1) for _ in range(N)]
        packet_delays = []

        def node_process(env, node_id, queue, server):
            while True:
                packet = yield queue.get()
                with server.request() as req:
                    yield req
                    # Service time
                    yield env.timeout(random.expovariate(mu))
                    
                    # 1. Attack discovery (After Service)
                    if random.random() < p:
                        new_p = NetworkPacket(packet.packet_id + "_retx", packet.entry_node, env.now)
                        new_p.original_arrival_time = packet.original_arrival_time
                        new_p.attempt_start_time = env.now
                        queues[packet.entry_node].put(new_p)
                        continue
                        
                    # 2. Timeout check (After Service)
                    if (env.now - packet.attempt_start_time) > W:
                        new_p = NetworkPacket(packet.packet_id + "_timeout", packet.entry_node, env.now)
                        new_p.original_arrival_time = packet.original_arrival_time
                        new_p.attempt_start_time = env.now
                        queues[packet.entry_node].put(new_p)
                        continue
                        
                    # 3. Routing decision
                    rand_choice = random.random()
                    transition_prob = 1 / (N + 1)
                    if rand_choice < transition_prob:
                        # Exit system
                        packet_delays.append(env.now - packet.original_arrival_time)
                    else:
                        # Continue to random node
                        dest = random.randint(0, N-1)
                        packet.current_node = dest
                        queues[dest].put(packet)

        def packet_generator(env, node_id):
            packet_counter = 0
            while True:
                yield env.timeout(random.expovariate(lambda_arr))
                p_obj = NetworkPacket(f"N{node_id}P{packet_counter}", node_id, env.now)
                queues[node_id].put(p_obj)
                packet_counter += 1

        for i in range(N):
            env.process(packet_generator(env, i))
            env.process(node_process(env, i, queues[i], servers[i]))

        env.run(until=warmup_period)
        packet_delays.clear()
        env.run(until=warmup_period + sim_duration)
        return np.mean(packet_delays) if packet_delays else np.inf
    return sim_func

def run_multiple_simulations(N, mu, lambda_arr, p, W, replications):
    results = []
    for i in range(replications):
        random.seed(i)
        np.random.seed(i)
        results.append(run_simulation(N, mu, lambda_arr, p, W)())
    return np.mean(results)

def plot_sojourn_time_vs_N_varying_p():
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'purple']
    all_results = []
    for idx, p in enumerate(p_values):
        theory_times, sim_times = [], []
        for N in N_values:
            print(f"Processing N={N}, p={p}...")
            avg_sojourn, _, _ = solve_feedback_network_theory(N, mu, lambda_arrival, p, W)
            theory_times.append(avg_sojourn if avg_sojourn is not None else np.inf)
            sim_val = run_multiple_simulations(N, mu, lambda_arrival, p, W, replications)
            sim_times.append(sim_val)
            all_results.append({"N": int(N), "p": float(p), "theory": float(theory_times[-1]) if theory_times[-1] != np.inf else "inf", "sim": float(sim_times[-1]) if sim_times[-1] != np.inf else "inf"})
            print(f"  Result: Theory={theory_times[-1]:.4f}, Sim={sim_times[-1]:.4f}")
        
        valid = [i for i, t in enumerate(theory_times) if t != np.inf]
        if valid:
            v_N = [list(N_values)[i] for i in valid]
            plt.plot(v_N, [theory_times[i] for i in valid], color=colors[idx], linestyle='-', label=f'Theory p={p}', linewidth=2)
            plt.scatter(v_N, [sim_times[i] for i in valid], color=colors[idx], marker='x', s=100, label=f'Simulation p={p}')
            
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Average Sojourn Time')
    plt.title('Figure 3.9: Feedback Network - Average Sojourn Time vs N')
    plt.legend()
    plt.grid(True)
    plt.savefig('section_3_3_1_sojourn_vs_N_varying_p.png')
    return all_results

if __name__ == "__main__":
    results = plot_sojourn_time_vs_N_varying_p()
    with open('results_feedback.json', 'w') as f:
        json.dump(results, f, indent=4)
