import numpy as np
import matplotlib.pyplot as plt
import simpy
import random

# --- MODEL PARAMETERS ---
mu = 10
lambda_a = 1.5
T = 30 # Using a more sensitive timeout
normal_traffic_rates = np.linspace(3, 5, 6)
attack_effectiveness_values = [0.2, 0.5]

# --- SIMULATION PARAMETERS ---
replications = 50
warmup_period = 500
sim_duration = 2000 # Using a longer duration for stability

# --- NEW CORRECTED THEORETICAL MODEL ---
def calculate_theoretic_delay_final(lambda_n, a):
    """
    Calculates the theoretical delay using the specific utilization formula
    from the user's paper (Eq. 3.13).
    """
    Lambda_star_old = lambda_n
    for _ in range(100):
        # Calculate Packet Loss (L) using Eq. 3.14 from the text
        if (a * lambda_a) < Lambda_star_old:
            L = (a * lambda_a) / (mu + lambda_a)
        else:
            L = Lambda_star_old / (mu + lambda_a)

        # *** KEY MODIFICATION ***
        # Calculate utilization (rho) using the specific formula from the paper (Eq. 3.13)
        utilization = Lambda_star_old / (mu + lambda_a)
        
        # Calculate Average Waiting Time E[W] using the standard M/M/1 formula
        # E[W] = 1 / (mu * (1 - rho)). This assumes W is total time in system.
        # Let's use the more common E[W] = 1/(mu-load) form.
        # Load = mu * rho.
        load = mu * utilization
        if load >= mu:
             EW = np.inf
        else:
             EW = 1 / (mu - load)
            
        # The retransmission probability (q) calculation remains the same
        P_W_gt_T = np.exp(-T / EW) if EW != np.inf and EW > 0 else 1
        q = L + (1 - L) * P_W_gt_T

        if q >= 1:
            Lambda_star_new = mu
        else:
            Lambda_star_new = lambda_n / (1 - q)
        
        if np.isclose(Lambda_star_new, Lambda_star_old, atol=1e-6) or Lambda_star_new >= mu:
            break
        Lambda_star_old = Lambda_star_new
    
    # Final calculation must also use the final values
    final_utilization = Lambda_star_new / (mu + lambda_a)
    final_load = mu * final_utilization
    
    final_EW = 1 / (mu - final_load) if final_load < mu else np.inf
    return final_EW

# --- SIMULATION MODEL (Unchanged) ---

packet_delays = []

class Packet:
    def __init__(self, identifier, arrival_time):
        self.identifier, self.original_arrival_time, self.current_arrival_time = identifier, arrival_time, arrival_time
        self.is_destroyed = False

def packet_generator(env, queue, lambda_n):
    packet_id = 0
    while True:
        yield env.timeout(random.expovariate(lambda_n))
        yield queue.put(Packet(f"Packet_{packet_id}", env.now))
        packet_id += 1

def attacker_destroy(env, queue, lambda_a, a):
    while True:
        yield env.timeout(random.expovariate(lambda_a))
        if queue.items and random.random() < a:
            random.choice(queue.items).is_destroyed = True

def server_process_destroy(env, server, queue):
    while True:
        packet = yield queue.get()
        if packet.is_destroyed:
            packet.is_destroyed, packet.current_arrival_time = False, env.now
            yield queue.put(packet)
            continue
        with server.request() as req:
            yield req
            waiting_time = env.now - packet.current_arrival_time
            yield env.timeout(random.expovariate(mu))
            if waiting_time > T:
                #print(f"Waiting time: {waiting_time}")

                packet.current_arrival_time = env.now
                yield queue.put(packet)
            else:
                packet_delays.append(env.now - packet.original_arrival_time)

def run_true_simulation(lambda_n, a, warmup_period, sim_duration):    
    global packet_delays
    packet_delays = []
    env = simpy.Environment()
    packet_queue, server = simpy.Store(env), simpy.Resource(env, capacity=1)
    env.process(packet_generator(env, packet_queue, lambda_n))
    env.process(attacker_destroy(env, packet_queue, lambda_a, a))
    env.process(server_process_destroy(env, server, packet_queue))
    env.run(until=warmup_period)
    packet_delays = []
    env.run(until=warmup_period + sim_duration)
    return np.mean(packet_delays) if packet_delays else np.inf

def run_multiple_simulations(lambda_n, a, replications, warmup_period, sim_duration):
    print(f"  Simulating with λn={lambda_n:.2f}...")
    
    replication_results = []
    for i in range(replications):
        random.seed(i); np.random.seed(i)
        avg_delay = run_true_simulation(lambda_n, a, warmup_period, sim_duration)
        if avg_delay != np.inf: replication_results.append(avg_delay)
    return np.mean(replication_results) if replication_results else np.inf

# --- EXECUTION AND PLOTTING ---

theoretic_results, simulation_results = {}, {}
for a in attack_effectiveness_values:
    theoretic_results[a] = [calculate_theoretic_delay_final(ln, a) for ln in normal_traffic_rates]
    print(f"\nRunning 'Destroy' simulations for a={a}...")
    simulated_delays = [run_multiple_simulations(ln, a, replications, warmup_period, sim_duration) if theoretic_results[a][list(normal_traffic_rates).index(ln)] != np.inf else np.inf for ln in normal_traffic_rates]
    simulation_results[a] = simulated_delays
    print("...done.")

plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r']
for idx, a in enumerate(attack_effectiveness_values):
    plt.plot(normal_traffic_rates, theoretic_results[a], color=colors[idx], linestyle='-', label=f'Theoretic (destroy) a={a}')
    plt.scatter(normal_traffic_rates, simulation_results[a], color=colors[idx], marker='x', s=100, label=f'Simulated (destroy) a={a}')

plt.xlabel('Normal Traffic Rate (λn)'); plt.ylabel('Average Packet Delay (s)')
plt.title('Corrected Theoretic vs. Simulated Delay for Attacks that Destroy Packets')
plt.legend(); plt.grid(True)
filename = f"plot_destroy_T{T}_reps{replications}_warmup{warmup_period}_sim{sim_duration}.png"
plt.savefig(filename)
print(f"\nPlot saved as {filename}")
