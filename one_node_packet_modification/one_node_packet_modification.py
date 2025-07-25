import numpy as np
import matplotlib.pyplot as plt
import simpy
import random

# --- THEORETICAL MODEL (Unchanged) ---
# Parameters for both theory and simulation
mu = 10
lambda_a = 1.5
T = 2
normal_traffic_rates = np.linspace(1, 5, 10)
attack_effectiveness_values = [0.2, 0.5, 0.8]

replications = 50
warmup_period = 500
sim_duration = 2000

def calculate_theoretic_delay(lambda_n, a):
    """Calculates the theoretical average delay based on the provided equations."""
    Lambda_star_old = lambda_n
    for _ in range(100):
        if Lambda_star_old >= mu:
            EW = np.inf
        else:
            EW = 1 / (mu - Lambda_star_old)
        
        L = (a * lambda_a) / mu if a * lambda_a < Lambda_star_old else Lambda_star_old / mu
        P_W_gt_T = np.exp(-T / EW) if EW != np.inf and EW > 0 else 1
        q = L + (1 - L) * P_W_gt_T

        if q >= 1:
            Lambda_star_new = mu
        else:
            Lambda_star_new = lambda_n / (1 - q)
        
        if np.isclose(Lambda_star_new, Lambda_star_old, atol=1e-6) or Lambda_star_new >= mu:
            break
        Lambda_star_old = Lambda_star_new
    
    final_EW = 1 / (mu - Lambda_star_new) if Lambda_star_new < mu else np.inf
    return final_EW

# --- SIMULATION MODEL ---

packet_delays = []

class Packet:
    """A class to represent packets, tracking their state."""
    def __init__(self, identifier, arrival_time):
        self.identifier = identifier
        self.original_arrival_time = arrival_time
        self.current_arrival_time = arrival_time
        self.corrupted = False

def packet_generator(env, queue, lambda_n):
    """Generates packets with a Poisson arrival process."""
    packet_id = 0
    while True:
        yield env.timeout(random.expovariate(lambda_n))
        packet = Packet(f"Packet_{packet_id}", env.now)
        yield queue.put(packet)
        packet_id += 1

def attacker(env, queue, lambda_a, a):
    """Generates attacks that can corrupt packets in the queue."""
    while True:
        yield env.timeout(random.expovariate(lambda_a))
        if queue.items and random.random() < a:
            target_packet = random.choice(queue.items)
            target_packet.corrupted = True

def server_process(env, server, queue):
    """Models the server processing packets from the queue."""
    while True:
        packet = yield queue.get()
        
        with server.request() as req:
            yield req
            
            waiting_time = env.now - packet.current_arrival_time
            yield env.timeout(random.expovariate(mu)) # Service time
            
            if packet.corrupted or waiting_time > T:
                # Retransmit if corrupted or timed out
                packet.corrupted = False
                packet.current_arrival_time = env.now
                yield queue.put(packet)
            else:
                # Success: record the total delay
                total_delay = env.now - packet.original_arrival_time
                packet_delays.append(total_delay)

def run_true_simulation(lambda_n, a):
    """Runs a single simulation instance for a given set of parameters."""
    global packet_delays
    packet_delays = []
    
    env = simpy.Environment()
    packet_queue = simpy.Store(env)
    server = simpy.Resource(env, capacity=1)
    
    env.process(packet_generator(env, packet_queue, lambda_n))
    env.process(attacker(env, packet_queue, lambda_a, a))
    env.process(server_process(env, server, packet_queue))
    
    env.run(until=warmup_period)
    packet_delays = []
    env.run(until=warmup_period + sim_duration)

    return np.mean(packet_delays) if packet_delays else np.inf

# --- NEW: FUNCTION FOR MULTIPLE REPLICATIONS ---
def run_multiple_simulations(lambda_n, a, replications):
    """
    Runs the simulation multiple times (replications) and averages the results
    to get a more statistically stable estimate of the average delay.
    """
    replication_results = []
    for i in range(replications):
        # Set a different seed for each replication for statistical independence
        random.seed(i) 
        np.random.seed(i)
        
        avg_delay = run_true_simulation(lambda_n, a)
        if avg_delay != np.inf:
            replication_results.append(avg_delay)
    
    # Return the average of all successful replications
    return np.mean(replication_results) if replication_results else np.inf

# --- EXECUTION AND PLOTTING ---
theoretic_results = {}
simulation_results = {}

for a in attack_effectiveness_values:
    # Calculate theoretical results
    theoretic_results[a] = [calculate_theoretic_delay(ln, a) for ln in normal_traffic_rates]
    
    print(f"\nRunning simulations for attack effectiveness a={a}...")
    simulated_delays = []
    for ln in normal_traffic_rates:
        print(f"  Simulating with λn={ln:.2f}...")
        # Check if the system is theoretically unstable first
        if theoretic_results[a][list(normal_traffic_rates).index(ln)] == np.inf:
            simulated_delays.append(np.inf)
        else:
            # MODIFIED: Call the function to run multiple replications
            avg_delay_from_replications = run_multiple_simulations(ln, a, replications)
            simulated_delays.append(avg_delay_from_replications)
            
    simulation_results[a] = simulated_delays
    print("...done.")

# Plotting the results
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'y', 'm']
for idx, a in enumerate(attack_effectiveness_values):
    plt.plot(normal_traffic_rates, theoretic_results[a], color=colors[idx], linestyle='-', label=f'Theoretic delay a={a}')
    plt.scatter(normal_traffic_rates, simulation_results[a], color=colors[idx], marker='x', s=100, label=f'Simulated delay a={a}')

plt.xlabel('Normal Traffic Rate (λn)')
plt.ylabel('Average Packet Delay (s)')
plt.title('Comparison of Theoretic and Simulated Delay (with Multiple Replications)')
plt.legend()
plt.grid(True)
filename = f"plot_reps{replications}_warmup{warmup_period}_sim{sim_duration}.png"
plt.savefig(filename)

print(f"\nPlot saved as {filename}")