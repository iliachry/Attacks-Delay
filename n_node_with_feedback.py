import numpy as np
import matplotlib.pyplot as plt
import simpy
import random
from scipy.optimize import fsolve

# --- MODEL PARAMETERS FOR SECTION 3.3.1 ---
mu = 1.0  # Service rate (identical for all nodes)
lambda_arrival = 0.2  # Arrival rate (identical for all nodes) 
W = 5.0  # Timeout period
N_values = range(2, 11)  # Number of nodes to test (2 to 10)
p_values = [0.1, 0.2, 0.3, 0.4]  # Attack probabilities to test
lambda_values = [0.1, 0.2, 0.3, 0.4]  # Arrival rates to test

# Simulation parameters
replications = 30
warmup_period = 1000
sim_duration = 5000

# --- THEORETICAL MODEL FOR SECTION 3.3.1 ---

def solve_feedback_network_theory(N, mu, lambda_arr, p, W):
    """
    Solves the theoretical model for N-Node feedback network from Section 3.3.1.
    
    Parameters:
    N: Number of nodes
    mu: Service rate (same for all nodes)
    lambda_arr: Arrival rate (same for all nodes)  
    p: Attack probability (same for all nodes)
    W: Timeout period
    """
    
    # Transition probabilities: equiprobable transitions Pij = 1/(N+1)
    P = np.ones((N, N+1)) / (N + 1)  # Include exit node N+1
    
    # Initial guess for variables
    initial_guess = np.concatenate([
        np.ones(N) * 0.1,  # L_i (loss probabilities)
        np.ones(N) * 1.0,  # T_i (sojourn times)
        np.ones(N) * lambda_arr * 1.5  # Lambda*_i (total traffic rates)
    ])
    
    def equations(vars):
        L = vars[:N]  # Loss probabilities
        T = vars[N:2*N]  # Sojourn times  
        Lambda_star = vars[2*N:3*N]  # Total traffic rates
        
        equations_list = []
        
        # Equation 3.21: Loss probability equations
        for i in range(N):
            eq = L[i] - (p + (1 - p) * sum(P[i, j] * L[j] for j in range(N)))
            equations_list.append(eq)
        
        # Equation 3.22: Sojourn time equations
        for i in range(N):
            if Lambda_star[i] >= mu:
                eq = T[i] - np.inf  # Unstable system
            else:
                eq = T[i] - (1/(mu - Lambda_star[i]) + sum(P[i, j] * T[j] for j in range(N)))
            equations_list.append(eq)
        
        # Equation 3.25: Total traffic rate equations  
        for i in range(N):
            lambda_star_i = lambda_arr / ((1 - np.exp(-W/T[i])) * (1 - L[i])) if T[i] > 0 and L[i] < 1 else np.inf
            
            total_from_others = sum(Lambda_star[j] * (1 - p) * P[j, i] for j in range(N))
            eq = Lambda_star[i] - (lambda_star_i + total_from_others)
            equations_list.append(eq)
        
        return equations_list
    
    try:
        # Solve the system of equations
        solution = fsolve(equations, initial_guess, xtol=1e-10)
        
        L = solution[:N]
        T = solution[N:2*N]  
        Lambda_star = solution[2*N:3*N]
        
        # Check for convergence and stability
        if any(Lambda_star >= mu * 0.99) or any(T <= 0) or any(L >= 1) or any(L < 0):
            return None, None, None
        
        # Calculate average sojourn time and total traffic rate
        avg_sojourn_time = np.mean(T)
        avg_total_traffic = np.mean(Lambda_star)
        
        return avg_sojourn_time, avg_total_traffic, (L, T, Lambda_star)
    
    except:
        return None, None, None

# --- SIMULATION MODEL FOR SECTION 3.3.1 ---

class NetworkPacket:
    def __init__(self, packet_id, entry_node, arrival_time):
        self.packet_id = packet_id
        self.entry_node = entry_node
        self.original_arrival_time = arrival_time
        self.current_node = entry_node
        self.path_history = [entry_node]

def create_feedback_network_simulation(N, mu, lambda_arr, p, W):
    """Creates a simulation of the feedback network from Section 3.3.1."""
    
    packet_delays = []
    
    def packet_generator(env, node_id):
        """Generate packets entering at node_id."""
        packet_counter = 0
        while True:
            yield env.timeout(random.expovariate(lambda_arr))
            packet = NetworkPacket(f"N{node_id}P{packet_counter}", node_id, env.now)
            queues[node_id].put(packet)
            packet_counter += 1
    
    def node_process(env, node_id, queue, server):
        """Process packets at a node with feedback routing."""
        while True:
            packet = yield queue.get()
            
            # Check for attack at this node
            if random.random() < p:
                # Packet lost due to attack - retransmit from entry node
                new_packet = NetworkPacket(packet.packet_id + "_retx", packet.entry_node, env.now)
                new_packet.original_arrival_time = packet.original_arrival_time
                queues[packet.entry_node].put(new_packet)
                continue
            
            with server.request() as req:
                yield req
                
                # Service time
                yield env.timeout(random.expovariate(mu))
                
                # Check timeout
                total_time = env.now - packet.original_arrival_time
                if total_time > W:
                    # Timeout - retransmit from entry node
                    new_packet = NetworkPacket(packet.packet_id + "_timeout", packet.entry_node, env.now)
                    new_packet.original_arrival_time = packet.original_arrival_time
                    queues[packet.entry_node].put(new_packet)
                    continue
                
                # Routing decision (equiprobable)
                rand_choice = random.random()
                cumulative_prob = 0
                
                # Can go to any node (including feedback) or exit
                transition_prob = 1 / (N + 1)
                
                destination = None
                for next_node in range(N):
                    cumulative_prob += transition_prob
                    if rand_choice < cumulative_prob:
                        destination = next_node
                        break
                
                if destination is None:
                    # Exit the network - success!
                    final_delay = env.now - packet.original_arrival_time
                    packet_delays.append(final_delay)
                else:
                    # Route to next node
                    packet.current_node = destination
                    packet.path_history.append(destination)
                    queues[destination].put(packet)
    
    def run_simulation():
        nonlocal packet_delays
        packet_delays = []
        
        env = simpy.Environment()
        
        # Create queues and servers for each node
        global queues, servers
        queues = [simpy.Store(env) for _ in range(N)]
        servers = [simpy.Resource(env, capacity=1) for _ in range(N)]
        
        # Start processes
        for i in range(N):
            env.process(packet_generator(env, i))
            env.process(node_process(env, i, queues[i], servers[i]))
        
        # Run simulation
        env.run(until=warmup_period)
        packet_delays = []  # Reset after warmup
        env.run(until=warmup_period + sim_duration)
        
        return np.mean(packet_delays) if packet_delays else np.inf
    
    return run_simulation

def run_multiple_simulations(N, mu, lambda_arr, p, W, replications):
    """Run multiple simulation replications."""
    sim_func = create_feedback_network_simulation(N, mu, lambda_arr, p, W)
    
    results = []
    for i in range(replications):
        random.seed(i)
        np.random.seed(i)
        result = sim_func()
        if result != np.inf and not np.isnan(result):
            results.append(result)
    
    return np.mean(results) if results else np.inf

# --- EXECUTION AND PLOTTING ---

def plot_sojourn_time_vs_N_varying_p():
    """Reproduce Figure 3.9: Average Sojourn time vs N, varying attack probability."""
    
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'purple']
    
    for idx, p in enumerate(p_values):
        theory_times = []
        sim_times = []
        
        for N in N_values:
            print(f"Processing N={N}, p={p}...")
            
            # Theoretical calculation
            avg_sojourn, avg_traffic, details = solve_feedback_network_theory(N, mu, lambda_arrival, p, W)
            theory_times.append(avg_sojourn if avg_sojourn is not None else np.inf)
            
            # Simulation
            if avg_sojourn is not None and avg_sojourn != np.inf:
                sim_result = run_multiple_simulations(N, mu, lambda_arrival, p, W, replications)
                sim_times.append(sim_result)
            else:
                sim_times.append(np.inf)
        
        # Plot results
        plt.plot(N_values, theory_times, color=colors[idx], linestyle='-', 
                label=f'Theory p={p}', linewidth=2)
        plt.scatter(N_values, sim_times, color=colors[idx], marker='x', s=100,
                   label=f'Simulation p={p}')
    
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Average Sojourn Time')
    plt.title('Section 3.3.1: Feedback Network - Average Sojourn Time vs N (Varying Attack Probability)')
    plt.legend()
    plt.grid(True)
    plt.savefig('section_3_3_1_sojourn_vs_N_varying_p.png')
    plt.show()

def plot_sojourn_time_vs_N_varying_lambda():
    """Reproduce Figure 3.10: Average Sojourn time vs N, varying arrival rate."""
    
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'purple']
    fixed_p = 0.3
    
    for idx, lam in enumerate(lambda_values):
        theory_times = []
        sim_times = []
        
        for N in N_values:
            print(f"Processing N={N}, λ={lam}...")
            
            # Theoretical calculation
            avg_sojourn, avg_traffic, details = solve_feedback_network_theory(N, mu, lam, fixed_p, W)
            theory_times.append(avg_sojourn if avg_sojourn is not None else np.inf)
            
            # Simulation  
            if avg_sojourn is not None and avg_sojourn != np.inf:
                sim_result = run_multiple_simulations(N, mu, lam, fixed_p, W, replications)
                sim_times.append(sim_result)
            else:
                sim_times.append(np.inf)
        
        # Plot results
        plt.plot(N_values, theory_times, color=colors[idx], linestyle='-',
                label=f'Theory λ={lam}', linewidth=2)
        plt.scatter(N_values, sim_times, color=colors[idx], marker='x', s=100,
                   label=f'Simulation λ={lam}')
    
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Average Sojourn Time') 
    plt.title('Section 3.3.1: Feedback Network - Average Sojourn Time vs N (Varying Arrival Rate)')
    plt.legend()
    plt.grid(True)
    plt.savefig('section_3_3_1_sojourn_vs_N_varying_lambda.png')
    plt.show()

def plot_traffic_rate_analysis():
    """Generate traffic rate analysis plots (Figures 3.11 and 3.12)."""
    
    # Figure 3.11: Total Traffic Rate vs N, varying attack probability
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'purple']
    
    for idx, p in enumerate(p_values):
        traffic_rates = []
        
        for N in N_values:
            avg_sojourn, avg_traffic, details = solve_feedback_network_theory(N, mu, lambda_arrival, p, W)
            traffic_rates.append(avg_traffic if avg_traffic is not None else np.inf)
        
        plt.plot(N_values, traffic_rates, color=colors[idx], linestyle='-', marker='o',
                label=f'p={p}', linewidth=2, markersize=6)
    
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Total Traffic Rate')
    plt.title('Section 3.3.1: Feedback Network - Total Traffic Rate vs N (Varying Attack Probability)')
    plt.legend()
    plt.grid(True)
    plt.savefig('section_3_3_1_traffic_vs_N_varying_p.png')
    plt.show()
    
    # Figure 3.12: Total Traffic Rate vs N, varying arrival rate
    plt.figure(figsize=(12, 8))
    fixed_p = 0.3
    
    for idx, lam in enumerate(lambda_values):
        traffic_rates = []
        
        for N in N_values:
            avg_sojourn, avg_traffic, details = solve_feedback_network_theory(N, mu, lam, fixed_p, W)
            traffic_rates.append(avg_traffic if avg_traffic is not None else np.inf)
        
        plt.plot(N_values, traffic_rates, color=colors[idx], linestyle='-', marker='o',
                label=f'λ={lam}', linewidth=2, markersize=6)
    
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Total Traffic Rate')
    plt.title('Section 3.3.1: Feedback Network - Total Traffic Rate vs N (Varying Arrival Rate)')
    plt.legend()
    plt.grid(True)
    plt.savefig('section_3_3_1_traffic_vs_N_varying_lambda.png')
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Section 3.3.1: N-Node Network with Feedback under Attacks")
    print("="*60)
    
    print("\nGenerating Figure 3.9: Average Sojourn Time vs N (Varying Attack Probability)...")
    plot_sojourn_time_vs_N_varying_p()
    
    print("\nGenerating Figure 3.10: Average Sojourn Time vs N (Varying Arrival Rate)...")  
    plot_sojourn_time_vs_N_varying_lambda()
    
    print("\nGenerating Traffic Rate Analysis (Figures 3.11 and 3.12)...")
    plot_traffic_rate_analysis()
    
    print("\nAll plots generated and saved!")
    
    # Example: Show detailed results for a specific case
    print(f"\nExample detailed results for N=5, μ={mu}, λ={lambda_arrival}, p={p_values[1]}, W={W}:")
    avg_sojourn, avg_traffic, details = solve_feedback_network_theory(5, mu, lambda_arrival, p_values[1], W)
    
    if details is not None:
        L, T, Lambda_star = details
        print(f"Average Sojourn Time: {avg_sojourn:.4f}")
        print(f"Average Total Traffic Rate: {avg_traffic:.4f}")
        print(f"Loss probabilities L_i: {L}")
        print(f"Sojourn times T_i: {T}")
        print(f"Total traffic rates Λ*_i: {Lambda_star}")
    else:
        print("System is unstable or solution did not converge.")
