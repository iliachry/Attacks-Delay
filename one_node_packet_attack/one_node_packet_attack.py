import numpy as np
import matplotlib.pyplot as plt
import simpy
import random

# --- MODEL PARAMETERS ---
mu = 10
lambda_a = 1.5
T = 2
normal_traffic_rates = np.linspace(1, 5, 10)
attack_effectiveness_values = [0.2, 0.5, 0.8]

# --- SIMULATION PARAMETERS ---
replications = 50
warmup_period = 500
sim_duration = 2000

packet_delays = []

# --- THEORETICAL MODEL (Unchanged) ---
def calculate_theoretic_delay_final(lambda_n, a):
    """Theoretical model for destruction attacks."""
    Lambda_star_old = lambda_n
    for _ in range(100):
        if (a * lambda_a) < Lambda_star_old:
            L = (a * lambda_a) / (mu + lambda_a)
        else:
            L = Lambda_star_old / (mu + lambda_a)

        if Lambda_star_old >= mu:
            EW = np.inf
        else:
            EW = 1 / (mu - Lambda_star_old)
            
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

# --- CORRECTED SIMULATION MODEL ---

class Packet:
    """A class to represent packets, tracking their state."""
    def __init__(self, identifier, arrival_time):
        self.identifier = identifier
        self.original_arrival_time = arrival_time
        self.current_arrival_time = arrival_time

def packet_generator(env, queue, lambda_n):
    """Generates packets with a Poisson arrival process."""
    packet_id = 0
    while True:
        yield env.timeout(random.expovariate(lambda_n))
        packet = Packet(f"Packet_{packet_id}", env.now)
        yield queue.put(packet)
        packet_id += 1

def attacker(env, queue, lambda_a, a):
    """Generates attacks that REMOVE packets from the queue (destruction)."""
    while True:
        yield env.timeout(random.expovariate(lambda_a))
        if queue.items and random.random() < a:
            # DESTRUCTION: Remove the packet entirely from the queue
            target_packet = random.choice(queue.items)
            queue.items.remove(target_packet)

            # Create a new packet for retransmission (immediate)
            retransmit_packet = Packet(target_packet.identifier + "_retx", env.now)
            retransmit_packet.original_arrival_time = target_packet.original_arrival_time
            retransmit_packet.current_arrival_time = target_packet.original_arrival_time
            yield queue.put(retransmit_packet)

def server_process(env, server, queue):
    """Models the server processing packets from the queue."""
    while True:
        packet = yield queue.get()
        
        with server.request() as req:
            yield req
            
            waiting_time = env.now - packet.current_arrival_time
            
            # Check timeout BEFORE service (destroyed packets never reach here)
            if waiting_time > T:
                # Timeout retransmission
                packet.current_arrival_time = env.now
                yield queue.put(packet)
            else:
                # Serve the packet (only non-destroyed packets reach here)
                yield env.timeout(random.expovariate(mu))  # Service time
                
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

def run_multiple_simulations(lambda_n, a, replications):
    """Runs multiple replications and averages results."""
    replication_results = []
    for i in range(replications):
        random.seed(i) 
        np.random.seed(i)
        
        avg_delay = run_true_simulation(lambda_n, a)
        if avg_delay != np.inf:
            replication_results.append(avg_delay)
    
    return np.mean(replication_results) if replication_results else np.inf

# --- EXECUTION AND PLOTTING ---
theoretic_results = {}
simulation_results = {}

for a in attack_effectiveness_values:
    theoretic_results[a] = [calculate_theoretic_delay_final(ln, a) for ln in normal_traffic_rates]
    
    print(f"\nRunning simulations for attack effectiveness a={a}...")
    simulated_delays = []
    for ln in normal_traffic_rates:
        print(f"  Simulating with λn={ln:.2f}...")
        if theoretic_results[a][list(normal_traffic_rates).index(ln)] == np.inf:
            simulated_delays.append(np.inf)
        else:
            avg_delay_from_replications = run_multiple_simulations(ln, a, replications)
            simulated_delays.append(avg_delay_from_replications)
            
    simulation_results[a] = simulated_delays
    print("...done.")

# Plotting
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'y', 'm']
for idx, a in enumerate(attack_effectiveness_values):
    plt.plot(normal_traffic_rates, theoretic_results[a], color=colors[idx], linestyle='-', 
             label=f'Theoretic delay (destroy) a={a}')
    plt.scatter(normal_traffic_rates, simulation_results[a], color=colors[idx], marker='x', s=100, 
                label=f'Simulated delay (destroy) a={a}')

plt.xlabel('Normal Traffic Rate (λn)')
plt.ylabel('Average Packet Delay (s)')
plt.title('Destruction Attacks: Packets Removed Before Service')
plt.legend()
plt.grid(True)
filename = f"destroy_no_service_plot_reps{replications}.png"
plt.savefig(filename)

print(f"\nPlot saved as {filename}")
