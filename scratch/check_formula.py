import numpy as np

mu = 10.0
B = 0.03
T = 2.0

def calc_theory_corrected(lambda_n, p):
    # Number of attempts
    E_A = 1.0 / (1.0 - p)
    # Server utilization is driven only by completed jobs
    rho = lambda_n / mu
    if rho >= 1.0:
        return np.inf
    # Queue waiting time in M/G/1 with service 0 (prob p) and Exp(mu) (prob 1-p)
    # E[W] = lambda_n / (mu * (mu - lambda_n))
    E_W = rho / (mu * (1.0 - rho))
    # Total delay = E[A] * E[W] + 1/mu + (E[A] - 1) * B
    return E_A * E_W + (1.0 / mu) + (E_A - 1.0) * B

# Compare with Sim values from previous test:
sim_vals = {
    0.2: [0.1196, 0.1261, 0.1334, 0.1417, 0.1509, 0.1615, 0.1738, 0.1881, 0.2051, 0.2259]
}

ln_vals = np.linspace(1, 5, 10)
for idx, ln in enumerate(ln_vals):
    th = calc_theory_corrected(ln, 0.2)
    sim = sim_vals[0.2][idx]
    err = abs(th - sim) / th * 100
    print(f"ln={ln:.2f} | Theory: {th:.4f}, Sim: {sim:.4f}, Err: {err:.2f}%")
