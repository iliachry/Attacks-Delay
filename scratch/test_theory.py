import numpy as np

def calculate_theoretic_delay_final(lambda_n, p, mu=10, T=2):
    """Robust fixed-point iteration for one-node delay with PRE-SERVICE destruction."""
    Lambda_star = lambda_n
    for i in range(200):
        if Lambda_star >= mu:
            return np.inf
        
        rho = (Lambda_star * (1 - p)) / mu
        if rho >= 1:
            return np.inf
            
        E_W = rho / (mu * (1 - rho)) 
        E_S = 1 / mu 
        E_D_attempt = E_W + (1 - p) * E_S 
        
        gamma = mu - (Lambda_star * (1 - p))
        P_succ = (1 - p) * (1 - np.exp(-gamma * T))
        
        if P_succ <= 0:
            return np.inf
            
        E_A = 1 / P_succ
        Lambda_star_new = lambda_n * E_A
        
        print(f"Iteration {i}: Lambda_star={Lambda_star:.4f}, Lambda_star_new={Lambda_star_new:.4f}, E_D_attempt={E_D_attempt:.4f}")
        
        if np.isclose(Lambda_star_new, Lambda_star, atol=1e-7):
            break
        Lambda_star = Lambda_star_new
        
    if Lambda_star >= mu:
        return np.inf
        
    return Lambda_star / lambda_n * E_D_attempt

print(f"Final Result (1.0, 0.2): {calculate_theoretic_delay_final(1.0, 0.2)}")
