import numpy as np
import matplotlib.pyplot as plt
import simpy
import random
from scipy.optimize import fsolve

# --- ADJUSTED MODEL PARAMETERS FOR SECTION 3.3.2 ---
mu = 2.0  # Increased service rate for better stability
lambda_arrival = 0.15  # Reduced arrival rate (was 0.3)
W = 8.0  # Increased timeout period for more realistic scenarios
N_values = range(2, 6)  # Reduced max nodes to avoid instability
p_values = [0.05, 0.1, 0.15, 0.2]  # Reduced attack probabilities
lambda_values = [0.05, 0.1, 0.15, 0.2]  # Reduced arrival rates

# Simulation parameters
replications = 30
warmup_period = 1000
sim_duration = 5000

# --- DEBUGGING FUNCTIONS ---

def debug_stability(N, mu, lambda_arr, p, W):
    """Debug which nodes cause instability."""
    print(f"\nDebugging N={N}, μ={mu}, λ={lambda_arr}, p={p}, W={W}")
    
    Lambda_star = np.zeros(N)
    T = np.zeros(N)
    L = np.zeros(N)
    
    # Calculate step by step
    for i in range(N):
        L[i] = p
        
        # Estimate traffic rate (simplified)
        if i == 0:
            Lambda_star[i] = lambda_arr / (1 - p)  # Rough estimate
        else:
            Lambda_star[i] = Lambda_star[i-1] * 1.2  # Estimate amplification
        
        print(f"  Node {i}: Λ*={Lambda_star[i]:.3f}, μ={mu}, ρ={Lambda_star[i]/mu:.3f}")
        
        if Lambda_star[i] >= mu:
            print(f"  *** UNSTABLE at node {i}: ρ={Lambda_star[i]/mu:.3f} ≥ 1 ***")
            return False
        
        T[i] = 1 / (mu - Lambda_star[i])
        print(f"  Node {i}: T={T[i]:.3f}")
    
    print("  System appears stable")
    return True

# --- ROBUST THEORETICAL MODEL FOR SECTION 3.3.2 ---

def solve_tandem_network_theory_robust(N, mu, lambda_arr, p, W):
    """
    Fixed solver for N-Node tandem network with proper retransmission logic.
    """
    
    # Pre-check: rough stability estimate
    max_amplification = (1 / (1 - p)) ** N  # Maximum possible amplification
    rough_estimate = lambda_arr * max_amplification
    if rough_estimate >= mu * 0.8:  # Less restrictive pre-check
        print(f"Pre-check failed: estimated load {rough_estimate:.3f} too high for μ={mu}")
        return None, None, None
    
    # Solver parameters
    max_iterations = 500
    tolerance = 1e-8
    damping_factor = 0.3  # More conservative damping
    
    # Initial guess: start with no retransmissions
    Lambda_star = np.full(N, lambda_arr * 1.1)
    
    for iteration in range(max_iterations):
        Lambda_star_old = Lambda_star.copy()
        
        # Calculate service times for all nodes
        T = np.zeros(N)
        for i in range(N):
            if Lambda_star[i] >= mu * 0.999:  # Stability check
                return None, None, None
            T[i] = 1 / (mu - Lambda_star[i])
        
        # Calculate end-to-end delay (only matters for final timeout)
        total_end_to_end_delay = np.sum(T)
        
        # Calculate probabilities
        L = np.full(N, p)  # Loss probability at each node
        # Timeout only applies to complete end-to-end journey
        P_timeout_final = 1 - np.exp(-W / total_end_to_end_delay)
        
        # For intermediate nodes, only loss matters. For final node, both loss and timeout
        Q = np.copy(L)  # Start with just loss probability
        Q[-1] = L[-1] + (1 - L[-1]) * P_timeout_final  # Final node gets timeout too
        
        # Check for instability
        if np.any(Q >= 0.999):
            return None, None, None
        
        # Update traffic rates
        # For tandem network with retransmissions from first node:
        # - First node gets new arrivals + all retransmissions
        # - Each subsequent node gets successful packets from previous node
        
        new_Lambda = np.zeros(N)
        
        # First node: new arrivals + retransmissions
        total_retransmission_rate = 0
        for j in range(N):
            # Packets that fail at node j and get retransmitted
            retransmission_rate = Lambda_star[j] * Q[j]
            total_retransmission_rate += retransmission_rate
        
        new_Lambda[0] = lambda_arr + total_retransmission_rate
        
        # Subsequent nodes: successful packets from previous node
        for i in range(1, N):
            # Packets successfully leaving node i-1
            successful_rate = Lambda_star[i-1] * (1 - Q[i-1])
            new_Lambda[i] = successful_rate
        
        # Apply damping for stability
        Lambda_star = damping_factor * new_Lambda + (1 - damping_factor) * Lambda_star
        
        # Check convergence
        if np.allclose(Lambda_star, Lambda_star_old, atol=tolerance):
            # Calculate final metrics
            T_final = np.array([1/(mu - ls) if ls < mu else np.inf for ls in Lambda_star])
            L_final = np.full(N, p)
            total_delay = np.sum(T_final)
            
            # Sanity check
            if np.any(np.isinf(T_final)) or np.isinf(total_delay):
                return None, None, None
                
            return total_delay, Lambda_star[0], (L_final, T_final, Lambda_star)
    
    print(f"No convergence after {max_iterations} iterations")
    return None, None, None

def solve_tandem_network_theory(N, mu, lambda_arr, p, W):
    """Wrapper function that calls the robust solver."""
    return solve_tandem_network_theory_robust(N, mu, lambda_arr, p, W)

# --- SIMULATION MODEL FOR SECTION 3.3.2 ---

class TandemPacket:
    def __init__(self, packet_id, arrival_time):
        self.packet_id = packet_id
        self.original_arrival_time = arrival_time
        self.current_node = 0
        self.path_history = [0]

def create_tandem_network_simulation(N, mu, lambda_arr, p, W):
    """Creates a simulation of the tandem network from Section 3.3.2."""
    
    packet_delays = []
    
    def packet_generator(env, first_queue):
        """Generate packets entering at first node only."""
        packet_counter = 0
        while True:
            yield env.timeout(random.expovariate(lambda_arr))
            packet = TandemPacket(f"P{packet_counter}", env.now)
            first_queue.put(packet)
            packet_counter += 1
    
    def tandem_node_process(env, node_id, queue, next_queue, server):
        """Process packets at a tandem node."""
        while True:
            packet = yield queue.get()
            
            # Check for attack at this node
            if random.random() < p:
                # Packet lost due to attack - retransmit from first node
                new_packet = TandemPacket(packet.packet_id + f"_retx_n{node_id}", env.now)
                new_packet.original_arrival_time = packet.original_arrival_time
                queues[0].put(new_packet)
                continue
            
            with server.request() as req:
                yield req
                
                # Service time
                yield env.timeout(random.expovariate(mu))
                
                # Check timeout (end-to-end)
                total_time = env.now - packet.original_arrival_time
                if total_time > W:
                    # Timeout - retransmit from first node
                    new_packet = TandemPacket(packet.packet_id + f"_timeout_n{node_id}", env.now)
                    new_packet.original_arrival_time = packet.original_arrival_time
                    queues[0].put(new_packet)
                    continue
                
                # Successful service at this node
                if node_id == N - 1:
                    # Last node - packet exits successfully
                    final_delay = env.now - packet.original_arrival_time
                    packet_delays.append(final_delay)
                else:
                    # Forward to next node in tandem
                    packet.current_node = node_id + 1
                    packet.path_history.append(node_id + 1)
                    next_queue.put(packet)
    
    def run_simulation():
        nonlocal packet_delays
        packet_delays = []
        
        env = simpy.Environment()
        
        # Create queues and servers for each node
        global queues, servers
        queues = [simpy.Store(env) for _ in range(N)]
        servers = [simpy.Resource(env, capacity=1) for _ in range(N)]
        
        # Start packet generator at first node only
        env.process(packet_generator(env, queues[0]))
        
        # Start node processes
        for i in range(N):
            next_queue = queues[i+1] if i < N-1 else None
            env.process(tandem_node_process(env, i, queues[i], next_queue, servers[i]))
        
        # Run simulation
        env.run(until=warmup_period)
        packet_delays = []  # Reset after warmup
        env.run(until=warmup_period + sim_duration)
        
        return np.mean(packet_delays) if packet_delays else np.inf
    
    return run_simulation

def run_multiple_simulations(N, mu, lambda_arr, p, W, replications):
    """Run multiple simulation replications."""
    sim_func = create_tandem_network_simulation(N, mu, lambda_arr, p, W)
    
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
    """Generate Figure for Section 3.3.2: Average Sojourn time vs N, varying attack probability."""
    
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'purple']
    
    for idx, p in enumerate(p_values):
        theory_times = []
        sim_times = []
        
        for N in N_values:
            print(f"Processing N={N}, p={p}...")
            
            # Check stability first
            if not debug_stability(N, mu, lambda_arrival, p, W):
                print(f"  Skipping unstable case N={N}, p={p}")
                theory_times.append(np.inf)
                sim_times.append(np.inf)
                continue
            
            # Theoretical calculation
            avg_sojourn, avg_traffic, details = solve_tandem_network_theory(N, mu, lambda_arrival, p, W)
            theory_times.append(avg_sojourn if avg_sojourn is not None else np.inf)
            
            # Simulation
            if avg_sojourn is not None and avg_sojourn != np.inf:
                sim_result = run_multiple_simulations(N, mu, lambda_arrival, p, W, replications)
                sim_times.append(sim_result)
            else:
                sim_times.append(np.inf)
        
        # Plot results (filter out infinite values for cleaner plots)
        valid_theory = [t if t != np.inf else None for t in theory_times]
        valid_sim = [s if s != np.inf else None for s in sim_times]
        
        plt.plot(N_values, valid_theory, color=colors[idx], linestyle='-', 
                label=f'Theory p={p}', linewidth=2)
        plt.scatter(N_values, valid_sim, color=colors[idx], marker='o', s=100,
                   label=f'Simulation p={p}')
    
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Average End-to-End Delay')
    plt.title('Section 3.3.2: Tandem Network - Average Delay vs N (Varying Attack Probability)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)  # Ensure positive y-axis
    plt.savefig('section_3_3_2_tandem_delay_vs_N_varying_p.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sojourn_time_vs_N_varying_lambda():
    """Generate Figure for Section 3.3.2: Average Sojourn time vs N, varying arrival rate."""
    
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'purple']
    fixed_p = 0.1  # Reduced from 0.2 for stability
    
    for idx, lam in enumerate(lambda_values):
        theory_times = []
        sim_times = []
        
        for N in N_values:
            print(f"Processing N={N}, λ={lam}...")
            
            # Check stability first
            if not debug_stability(N, mu, lam, fixed_p, W):
                print(f"  Skipping unstable case N={N}, λ={lam}")
                theory_times.append(np.inf)
                sim_times.append(np.inf)
                continue
            
            # Theoretical calculation
            avg_sojourn, avg_traffic, details = solve_tandem_network_theory(N, mu, lam, fixed_p, W)
            theory_times.append(avg_sojourn if avg_sojourn is not None else np.inf)
            
            # Simulation  
            if avg_sojourn is not None and avg_sojourn != np.inf:
                sim_result = run_multiple_simulations(N, mu, lam, fixed_p, W, replications)
                sim_times.append(sim_result)
            else:
                sim_times.append(np.inf)
        
        # Plot results (filter out infinite values)
        valid_theory = [t if t != np.inf else None for t in theory_times]
        valid_sim = [s if s != np.inf else None for s in sim_times]
        
        plt.plot(N_values, valid_theory, color=colors[idx], linestyle='-',
                label=f'Theory λ={lam}', linewidth=2)
        plt.scatter(N_values, valid_sim, color=colors[idx], marker='o', s=100,
                   label=f'Simulation λ={lam}')
    
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Average End-to-End Delay') 
    plt.title('Section 3.3.2: Tandem Network - Average Delay vs N (Varying Arrival Rate)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.savefig('section_3_3_2_tandem_delay_vs_N_varying_lambda.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_throughput_analysis():
    """Generate throughput analysis plots for Section 3.3.2."""
    
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'purple']
    
    for idx, p in enumerate(p_values):
        throughputs = []
        
        for N in N_values:
            avg_sojourn, avg_traffic, details = solve_tandem_network_theory(N, mu, lambda_arrival, p, W)
            
            if details is not None:
                L, T, Lambda_star = details
                # Throughput = successful rate at last node
                last_node_idx = -1
                if T[last_node_idx] != np.inf and T[last_node_idx] > 0:
                    throughput = Lambda_star[last_node_idx] * (1 - L[last_node_idx]) * (1 - np.exp(-W/T[last_node_idx]))
                else:
                    throughput = 0
            else:
                throughput = 0
                
            throughputs.append(throughput)
        
        plt.plot(N_values, throughputs, color=colors[idx], linestyle='-', marker='s',
                label=f'p={p}', linewidth=2, markersize=6)
    
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('System Throughput')
    plt.title('Section 3.3.2: Tandem Network - Throughput vs N (Varying Attack Probability)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.savefig('section_3_3_2_tandem_throughput_vs_N.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_stability_regions():
    """Generate stability region analysis."""
    
    plt.figure(figsize=(12, 8))
    
    # Create a grid of arrival rates and attack probabilities
    arrival_range = np.linspace(0.05, 0.3, 20)
    attack_range = np.linspace(0.05, 0.3, 20)
    
    stable_regions = {}
    
    for N in [2, 3, 4]:
        stability_matrix = np.zeros((len(attack_range), len(arrival_range)))
        
        for i, p in enumerate(attack_range):
            for j, lam in enumerate(arrival_range):
                result = solve_tandem_network_theory(N, mu, lam, p, W)
                stability_matrix[i, j] = 1 if result[0] is not None else 0
        
        stable_regions[N] = stability_matrix
    
    # Plot stability regions for different N values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, N in enumerate([2, 3, 4]):
        im = axes[idx].imshow(stable_regions[N], extent=[0.05, 0.3, 0.05, 0.3], 
                             origin='lower', cmap='RdYlGn', aspect='auto')
        axes[idx].set_xlabel('Arrival Rate (λ)')
        axes[idx].set_ylabel('Attack Probability (p)')
        axes[idx].set_title(f'Stability Region (N={N})')
        axes[idx].grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], label='Stable (1) / Unstable (0)')
    
    plt.tight_layout()
    plt.savefig('section_3_3_2_stability_regions.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Section 3.3.2: N-Node Tandem Network under Attacks (Robust Version)")
    print("="*70)
    
    # Test stability for a few cases first
    print("\nTesting stability for key parameter combinations...")
    test_cases = [
        (3, mu, lambda_arrival, 0.1, W),
        (4, mu, lambda_arrival, 0.15, W),
        (5, mu, lambda_arrival, 0.2, W)
    ]
    
    for case in test_cases:
        debug_stability(*case)
    
    print("\nGenerating Average Delay vs N (Varying Attack Probability)...")
    plot_sojourn_time_vs_N_varying_p()
    
    print("\nGenerating Average Delay vs N (Varying Arrival Rate)...")  
    plot_sojourn_time_vs_N_varying_lambda()
    
    print("\nGenerating Throughput Analysis...")
    plot_throughput_analysis()
    
    print("\nGenerating Stability Region Analysis...")
    plot_stability_regions()
    
    print("\nAll plots generated and saved!")
    
    # Example: Show detailed results for a specific case
    print(f"\nExample detailed results for N=3, μ={mu}, λ={lambda_arrival}, p={p_values[1]}, W={W}:")
    avg_sojourn, avg_traffic, details = solve_tandem_network_theory_robust(3, mu, lambda_arrival, p_values[1], W)
    
    if details is not None:
        L, T, Lambda_star = details
        print(f"Average End-to-End Delay: {avg_sojourn:.4f}")
        print(f"Total Input Traffic Rate: {avg_traffic:.4f}")
        print(f"Loss probabilities L_i: {L}")
        print(f"Sojourn times T_i: {T}")
        print(f"Traffic rates Λ*_i: {Lambda_star}")
        
        # Calculate system throughput
        if T[-1] != np.inf and T[-1] > 0:
            throughput = Lambda_star[-1] * (1 - L[-1]) * (1 - np.exp(-W/T[-1]))
        else:
            throughput = 0
        print(f"System Throughput: {throughput:.4f}")
    else:
        print("System is unstable or solution did not converge.")
