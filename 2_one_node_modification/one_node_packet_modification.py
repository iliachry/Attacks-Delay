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
sim_duration = 5000

def calculate_theoretic_delay(lambda_n, p):
    """Robust fixed-point iteration for one-node delay with retransmissions."""
    Lambda_star = lambda_n
    for _ in range(200):
        if Lambda_star >= mu:
            return np.inf
        
        # M/M/1 delay for one attempt
        E_S = 1 / (mu - Lambda_star)
        
        # Prob of success in one attempt: No attack and No timeout
        # P(S <= W) = 1 - exp(-W / E_S)
        P_succ = (1 - p) * (1 - np.exp(-T / E_S))
        
        if P_succ <= 0:
            return np.inf
            
        # Expected number of attempts
        E_A = 1 / P_succ
        
        # New effective traffic rate
        Lambda_star_new = lambda_n * E_A
        
        if np.isclose(Lambda_star_new, Lambda_star, atol=1e-7):
            break
        Lambda_star = Lambda_star_new
        
    if Lambda_star >= mu:
        return np.inf
        
    # Renewal theory delay: (E[A]-1)*E[D_fail] + E[D_succ]
    # Here E[D_fail] approx E[D_succ] approx E_S for simplicity in M/M/1
    # Full formula: E[D] = E[A] * E_S
    return E_A * (1 / (mu - Lambda_star))

# --- SIMULATION MODEL ---

packet_delays = []

class Packet:
    """A class to represent packets, tracking their state."""
    def __init__(self, identifier, arrival_time):
        self.identifier = identifier
        self.original_arrival_time = arrival_time
        self.attempt_start_time = arrival_time
        self.corrupted = False

def packet_generator(env, queue, lambda_n):
    """Generates packets with a Poisson arrival process."""
    packet_id = 0
    while True:
        yield env.timeout(random.expovariate(lambda_n))
        packet = Packet(f"Packet_{packet_id}", env.now)
        packet.attempt_start_time = env.now
        yield queue.put(packet)
        packet_id += 1

def server_process(env, server, queue, p):
    """Models the server processing packets from the queue."""
    while True:
        packet = yield queue.get()
        
        with server.request() as req:
            yield req
            
            # Note: attempt_start_time is ALREADY set when packet entered the queue
            
            # Discovery happens AFTER service
            yield env.timeout(random.expovariate(mu)) # Service time
            
            # Check for attack or timeout
            if random.random() < p or (env.now - packet.attempt_start_time) > T:
                # Retransmit: put back in queue
                packet.attempt_start_time = env.now # Reset timer for next attempt
                yield queue.put(packet)
            else:
                # Success: record total end-to-end delay
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
    env.process(server_process(env, server, packet_queue, a))
    
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
        print(f"  Simulating with lambda_n={ln:.2f}...")
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