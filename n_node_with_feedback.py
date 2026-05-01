import numpy as np
import matplotlib.pyplot as plt
import json
import simpy
import random
from scipy.optimize import fsolve

# --- MODEL PARAMETERS FOR SECTION 3.3.1 ---
mu = 1.0  # Service rate (identical for all nodes)
lambda_arrival = 0.1  # Reduced arrival rate for stability
W = 20.0  # Increased timeout period
N_values = range(2, 6)  # Number of nodes to test
p_values = [0.05, 0.1, 0.15, 0.2]  # Attack probabilities
lambda_values = [0.1, 0.2, 0.3, 0.4]  # Arrival rates to test

# Simulation parameters
replications = 30
warmup_period = 1000
sim_duration = 5000

# --- THEORETICAL MODEL FOR SECTION 3.3.1 ---

def solve_feedback_network_theory(N, mu, lambda_arr, p, W):
    """
    Solves the theoretical model for N-Node feedback network using a symmetric 
    approach and Erlang distribution for cumulative timeouts.
    """
    from scipy.special import gamma, gammainc
    
    # Erlang CDF: P(S_k <= W) = gammainc(k, gamma_rate * W)
    # Note: scipy.special.gammainc(a, x) is the regularized lower incomplete gamma function
    # which is exactly the CDF of Gamma(a, 1) at x. 
    # For Erlang(k, lambda), CDF is gammainc(k, lambda * W).
    
    def get_erlang_cdf(k, rate, w):
        if k <= 0: return 1.0
        return gammainc(k, rate * w)

    def equations(lambda_star_val):
        gamma_rate = mu - lambda_star_val
        if gamma_rate <= 0:
            return 1e6 # Large penalty for instability
            
        # We need to calculate E[V] and P_succ using infinite sums (truncated)
        p_succ = 0
        e_v = 0
        e_d_succ_num = 0
        e_d_fail_num = 0
        
        max_k = 100 # Sufficient for convergence in most stable cases
        
        for k in range(1, max_k):
            # Probability of reaching node k and continuing/succeeding
            # term = (1-p)^k * (N/(N+1))^(k-1)
            term = ((1 - p) ** k) * ((N / (N + 1)) ** (k - 1))
            
            f_k_minus_1 = get_erlang_cdf(k - 1, gamma_rate, W)
            f_k = get_erlang_cdf(k, gamma_rate, W)
            f_k_plus_1 = get_erlang_cdf(k + 1, gamma_rate, W)
            
            # P_succ: No attack in k visits, Continued k-1 times, Route to exit at k, No timeout at k
            p_succ_k = term * (1 / (N + 1)) * f_k
            p_succ += p_succ_k
            
            # E[V]: sum P(Visits >= k)
            e_v += term * f_k_minus_1
            
            # For delay:
            # Succ delay contribution: k/gamma * F(k+1)
            e_d_succ_num += term * (1 / (N + 1)) * (k / gamma_rate) * f_k_plus_1
            
            # Fail delay contribution:
            # 1. Attack at k: prob (1-p)^(k-1) * p * (N/(N+1))^(k-1) * F(k-1). Delay S_{k-1} = (k-1)/gamma
            p_attack_k = ((1-p)**(k-1)) * p * ((N/(N+1))**(k-1)) * f_k_minus_1
            e_d_fail_num += p_attack_k * ((k-1) / gamma_rate)
            
            # 2. Timeout at k: prob term * (N/(N+1) + 1/(N+1)) * (F(k-1) - F(k)). Delay S_k = k/gamma
            # Note: (N/(N+1) + 1/(N+1)) = 1.
            p_timeout_k = term * (f_k_minus_1 - f_k)
            e_d_fail_num += p_timeout_k * (k / gamma_rate)

        if p_succ <= 0:
            return lambda_star_val - 1e6
            
        # Fixed point equation: Lambda* = lambda * E[attempts] * E[visits_per_attempt]
        # E[attempts] = 1 / p_succ
        # E[visits_per_attempt] = e_v
        new_lambda_star = lambda_arr * (e_v / p_succ)
        
        # print(f"  DEBUG: lambda_star={lambda_star_val:.4f}, new={new_lambda_star:.4f}, p_succ={p_succ:.4f}")
        
        return lambda_star_val - new_lambda_star

    try:
        # Solve for Lambda* using fsolve or root finding
        from scipy.optimize import brentq
        
        # Check stability at low load
        f_low = equations(lambda_arr)
        if f_low > 0:
            # This would mean lambda_arr > lambda_arr * (e_v/p_succ), impossible
            return None, None, None
            
        # Check stability at high load
        f_high = equations(mu * 0.999)
        print(f"  DEBUG: N={N}, p={p}, f_low={f_low:.4f}, f_high={f_high:.4f}")
        if f_high < 0:
            # Unstable
            return None, None, None
            
        lambda_star_sol = brentq(equations, lambda_arr, mu * 0.999)
        
        gamma_rate = mu - lambda_star_sol
        
        # Re-calculate p_succ and other metrics for final delay
        p_succ = 0
        e_d_succ_num = 0
        e_d_fail_num = 0
        max_k = 100
        for k in range(1, max_k):
            term = ((1 - p) ** k) * ((N / (N + 1)) ** (k - 1))
            f_k_minus_1 = get_erlang_cdf(k - 1, gamma_rate, W)
            f_k = get_erlang_cdf(k, gamma_rate, W)
            f_k_plus_1 = get_erlang_cdf(k + 1, gamma_rate, W)
            
            p_succ += term * (1 / (N + 1)) * f_k
            e_d_succ_num += term * (1 / (N + 1)) * (k / gamma_rate) * f_k_plus_1
            
            p_attack_k = ((1-p)**(k-1)) * p * ((N/(N+1))**(k-1)) * f_k_minus_1
            e_d_fail_num += p_attack_k * ((k-1) / gamma_rate)
            
            p_timeout_k = term * (f_k_minus_1 - f_k)
            e_d_fail_num += p_timeout_k * (k / gamma_rate)
            
        e_attempts = 1 / p_succ
        e_d_succ = e_d_succ_num / p_succ
        e_d_fail = e_d_fail_num / (1 - p_succ) if p_succ < 1 else 0
        
        avg_sojourn_time = (e_attempts - 1) * e_d_fail + e_d_succ
        avg_total_traffic = lambda_star_sol
        
        # Symmetry: all nodes are the same
        L = np.full(N, 1 - p_succ)
        T = np.full(N, avg_sojourn_time)
        Lambda_star = np.full(N, lambda_star_sol)
        
        return avg_sojourn_time, avg_total_traffic, (L, T, Lambda_star)
        
    except Exception as e:
        # print(f"Error solving for N={N}: {e}")
        return None, None, None

# --- SIMULATION MODEL FOR SECTION 3.3.1 ---

class NetworkPacket:
    def __init__(self, packet_id, entry_node, arrival_time):
        self.packet_id = packet_id
        self.entry_node = entry_node
        self.original_arrival_time = arrival_time
        self.attempt_start_time = arrival_time  # New field for timeout checks
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
    
    
    def run_simulation():
        nonlocal packet_delays
        packet_delays = []
        
        env = simpy.Environment()
        
        # Create queues and servers for each node (local to this simulation)
        queues = [simpy.Store(env) for _ in range(N)]
        servers = [simpy.Resource(env, capacity=1) for _ in range(N)]
        
        # Local node process that uses local queues
        def node_process_local(env, node_id, queue, server):
            """Process packets at a node with feedback routing."""
            while True:
                packet = yield queue.get()
                
                if random.random() < p:
                    # Packet lost due to attack - retransmit from entry node
                    new_packet = NetworkPacket(packet.packet_id + "_retx", packet.entry_node, env.now)
                    new_packet.original_arrival_time = packet.original_arrival_time
                    new_packet.attempt_start_time = env.now # Timer resets for new attempt
                    queues[packet.entry_node].put(new_packet)
                    continue
                
                with server.request() as req:
                    yield req
                    
                    # Service time
                    yield env.timeout(random.expovariate(mu))
                    
                    # Check timeout
                    # Check timeout uses attempt_start_time
                    if (env.now - packet.attempt_start_time) > W:
                        # Timeout - retransmit from entry node
                        new_packet = NetworkPacket(packet.packet_id + "_timeout", packet.entry_node, env.now)
                        new_packet.original_arrival_time = packet.original_arrival_time
                        new_packet.attempt_start_time = env.now # Timer resets
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
                        queues[destination].put(packet)  # Use local queues
        
        # Local packet generator that uses local queues
        def packet_generator_local(env, node_id):
            """Generate packets entering at node_id."""
            packet_counter = 0
            while True:
                yield env.timeout(random.expovariate(lambda_arr))
                packet = NetworkPacket(f"N{node_id}P{packet_counter}", node_id, env.now)
                queues[node_id].put(packet)  # Use local queues
                packet_counter += 1
        
        # Start processes
        for i in range(N):
            env.process(packet_generator_local(env, i))
            env.process(node_process_local(env, i, queues[i], servers[i]))
        
        # Run simulation
        env.run(until=warmup_period)
        packet_delays = []  # Reset after warmup
        env.run(until=warmup_period + sim_duration)
        
        return np.mean(packet_delays) if packet_delays else np.inf
    
    return run_simulation

def run_multiple_simulations(N, mu, lambda_arr, p, W, replications):
    """Run multiple simulation replications."""
    results = []
    for i in range(replications):
        random.seed(i)
        np.random.seed(i)
        # Create a new simulation instance for each replication
        sim_func = create_feedback_network_simulation(N, mu, lambda_arr, p, W)
        result = sim_func()
        if result != np.inf and not np.isnan(result):
            results.append(result)
    
    return np.mean(results) if results else np.inf

# --- EXECUTION AND PLOTTING ---

    plt.grid(True)
    plt.savefig('section_3_3_1_sojourn_vs_N_varying_p.png')
    # plt.show()
    return all_results

def plot_sojourn_time_vs_N_varying_p():
    """Reproduce Figure 3.9: Average Sojourn time vs N, varying attack probability."""
    
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'purple']
    all_results = []
    
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
            
            all_results.append({
                "N": int(N),
                "p": float(p),
                "theory": float(theory_times[-1]) if theory_times[-1] != np.inf else "inf",
                "sim": float(sim_times[-1]) if sim_times[-1] != np.inf else "inf"
            })
            
            print(f"  Result: Theory={theory_times[-1]:.4f}, Sim={sim_times[-1]:.4f}, Gap={abs(theory_times[-1]-sim_times[-1]) if theory_times[-1] != np.inf and sim_times[-1] != np.inf else 0:.4f}")
        
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
    # plt.show()
    return all_results

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
    results = plot_sojourn_time_vs_N_varying_p()
    
    with open('results_feedback.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # plot_sojourn_time_vs_N_varying_lambda()
    
    # plot_traffic_rate_analysis()
    
    print("\nInitial results saved to results_feedback.json!")
    
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
