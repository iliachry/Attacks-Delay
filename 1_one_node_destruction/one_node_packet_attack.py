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
sim_duration = 5000

packet_delays = []

# --- THEORETICAL MODEL (Unchanged) ---
def calculate_theoretic_delay_final(lambda_n, p):
    """Robust fixed-point iteration for one-node delay with PRE-SERVICE destruction."""
    Lambda_star = lambda_n
    for _ in range(200):
        if Lambda_star >= mu:
            return np.inf
        
        # M/M/1 delay for one attempt
        # Traffic entering server is Lambda_star * (1 - p)
        rho = (Lambda_star * (1 - p)) / mu
        if rho >= 1:
            return np.inf
            
        E_W = rho / (mu * (1 - rho)) # Expected waiting time in queue
        E_S = 1 / mu # Service time
        E_D_attempt = E_W + (1 - p) * E_S # Average delay of one attempt
        
        # Prob of success in one attempt: No attack and No timeout
        # Success delay is W + S. CDF of W+S is harder, but for M/M/1 it is 1 - exp(- (mu-lambda) t)
        # Here effective mu is mu, effective lambda is Lambda_star * (1-p)
        gamma = mu - (Lambda_star * (1 - p))
        P_succ = (1 - p) * (1 - np.exp(-gamma * T))
        
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
        
    # Total delay = E[A] * E[D_attempt]
    return Lambda_star / lambda_n * E_D_attempt

# --- CORRECTED SIMULATION MODEL ---

class Packet:
    """A class to represent packets, tracking their state."""
    def __init__(self, identifier, arrival_time):
        self.identifier = identifier
        self.original_arrival_time = arrival_time
        self.attempt_start_time = arrival_time  # Tracks the start of the current attempt

def packet_generator(env, queue, lambda_n):
    """Generates packets with a Poisson arrival process."""
    packet_id = 0
    while True:
        yield env.timeout(random.expovariate(lambda_n))
        packet = Packet(f"Packet_{packet_id}", env.now)
        packet.attempt_start_time = env.now # Initial attempt start
        yield queue.put(packet)
        packet_id += 1

def server_process(env, server, queue, p):
    """Models the server processing packets from the queue."""
    while True:
        packet = yield queue.get()
        
        with server.request() as req:
            yield req
            
            # Note: attempt_start_time is ALREADY set when packet entered the queue
            
            # Pre-Service Attack check
            if random.random() < p:
                # Destruction: no service time consumed
                packet.attempt_start_time = env.now # Reset timer for next attempt
                yield queue.put(packet)
            else:
                # Service time consumed if not destroyed
                yield env.timeout(random.expovariate(mu))
                
                # Check for timeout (total time in node)
                if (env.now - packet.attempt_start_time) > T:
                    packet.attempt_start_time = env.now # Reset timer for next attempt
                    yield queue.put(packet)
                else:
                    # Success
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
        print(f"  Simulating with lambda_n={ln:.2f}...")
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
