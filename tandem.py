import numpy as np
import matplotlib.pyplot as plt
import simpy
import random

# --- MODEL PARAMETERS ---
N = 3
lambda_c_values = np.linspace(0.5, 3.5, 15)
mu_i = [10.0] * N
alpha_scenarios = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
lambda_ai = [1.0] * N
T = 30

# Simulation control
replications = 10
warmup_period = 500
sim_duration = 2000

# --- THEORETICAL MODEL (Unchanged) ---
def calculate_theoretic_delay(lambda_c, alpha_i_scenario):
    q = 0.0
    for _ in range(100):
        # This simplified equation assumes Λ* is the same for all nodes
        Lambda_star = lambda_c / (1 - q) if q < 1 else mu_i[0]
        
        # Check for stability
        if any(Lambda_star >= mu for mu in mu_i):
            total_delay = np.inf
            break

        p_success = np.prod([1 - (alpha_i_scenario[i] * lambda_ai[i] / Lambda_star) if Lambda_star > 0 else 1 for i in range(N)])
        L = 1 - p_success
        
        total_delay = sum([1 / (mu - Lambda_star) for mu in mu_i])
        
        P_W_gt_T = np.exp(-T / total_delay) if total_delay > 0 else 1.0
        q_new = L + (1 - L) * P_W_gt_T
        
        if np.isclose(q, q_new, atol=1e-8): break
        q = q_new
    
    # Recalculate final delay with converged q
    Lambda_star = lambda_c / (1 - q) if q < 1 else mu_i[0]
    if any(Lambda_star >= mu for mu in mu_i):
        return np.inf
    return sum([1 / (mu - Lambda_star) for mu in mu_i])


# --- NEW SIMULATION MODEL TO MATCH THE THEORY ---
packet_delays = []

class Packet:
    def __init__(self, id, arrival_time):
        self.id = id
        self.original_arrival_time = arrival_time
        self.path_start_time = arrival_time
        # This flag tracks if the packet was destroyed AT ANY point in its path
        self.was_destroyed_on_path = False

def packet_generator(env, lambda_c, first_queue):
    """Generates master packets arriving at the start of the connection."""
    pid = 0
    while True:
        yield env.timeout(random.expovariate(lambda_c))
        yield first_queue.put(Packet(f"P_{pid}", env.now))
        pid += 1

def attacker_process(env, alpha, lambda_a, queue):
    """Attacker for a single node. It just flags packets as destroyed."""
    if random.random() >= alpha: return
    
    while True:
        yield env.timeout(random.expovariate(lambda_a))
        if queue.items:
            target_packet = random.choice(queue.items)
            # Set the flag. The packet itself will not be removed from the queue.
            target_packet.was_destroyed_on_path = True

def node_process(env, mu, queue, next_queue, first_queue):
    """
    Models a node. It processes a packet and sends it to the next node,
    regardless of its 'destroyed' status.
    """
    server = simpy.Resource(env, capacity=1)
    while True:
        packet = yield queue.get()

        with server.request() as req:
            yield req
            yield env.timeout(random.expovariate(mu))

        is_last_node = (next_queue is None)
        if is_last_node:
            total_path_delay = env.now - packet.path_start_time
            # Retransmit if it was destroyed ANYWHERE or if it timed out
            if packet.was_destroyed_on_path or total_path_delay > T:
                # Reset for retransmission
                packet.path_start_time = env.now
                packet.was_destroyed_on_path = False
                yield first_queue.put(packet)
            else:
                # Success!
                final_delay = env.now - packet.original_arrival_time
                packet_delays.append(final_delay)
        else:
            # Move to the next node
            yield next_queue.put(packet)

def run_true_simulation(lambda_c, alpha_i_scenario):
    """Sets up and runs a simulation replication."""
    global packet_delays
    packet_delays = []
    env = simpy.Environment()

    queues = [simpy.Store(env) for _ in range(N)]
    for i in range(N):
        next_queue = queues[i+1] if i < N - 1 else None
        env.process(node_process(env, mu_i[i], queues[i], next_queue, queues[0]))
        env.process(attacker_process(env, alpha_i_scenario[i], lambda_ai[i], queues[i]))

    env.process(packet_generator(env, lambda_c, queues[0]))
    
    env.run(until=warmup_period)
    packet_delays = []
    env.run(until=warmup_period + sim_duration)
    
    return np.mean(packet_delays) if packet_delays else np.inf

def run_multiple_replications(lambda_c, alpha_i_scenario):
    results = [run_true_simulation(lambda_c, alpha_i_scenario) for i in range(replications)]
    valid_results = [r for r in results if r != np.inf]
    return np.mean(valid_results) if valid_results else np.inf

# --- EXECUTION & PLOTTING ---
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c']

for i, alpha_scenario in enumerate(alpha_scenarios):
    theoretic_delays = [calculate_theoretic_delay(lc, alpha_scenario) for lc in lambda_c_values]
    plt.plot(lambda_c_values, theoretic_delays, color=colors[i], linestyle='-', label=f'Theoretic α = {alpha_scenario}')
    
    print(f"Running simulation for alpha_i = {alpha_scenario}...")
    simulated_delays = [run_multiple_replications(lc, alpha_scenario) for lc in lambda_c_values]
    plt.scatter(lambda_c_values, simulated_delays, color=colors[i], marker='x', s=100, label=f'Simulated α = {alpha_scenario}')

plt.xlabel('Connection Traffic Rate (λc)'); plt.ylabel('Average Connection Delay (s)')
plt.title('Corrected Tandem Node Simulation vs. Theory')
plt.legend(); plt.grid(True); plt.ylim(bottom=0, top=1.5)
filename = f"tandem_simulation_corrected_N{N}.png"
plt.savefig(filename)
print(f"\nPlot saved as {filename}")

