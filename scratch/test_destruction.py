import numpy as np
import simpy
import random
from concurrent.futures import ProcessPoolExecutor

mu = 10.0
T = 2.0
replications = 50
sim_duration = 15000
warmup_period = 2000

def run_single(args):
    ln, p, seed = args
    random.seed(seed)
    np.random.seed(seed)
    delays = []
    env = simpy.Environment()
    q = simpy.Store(env)
    srv = simpy.Resource(env, capacity=1)
    
    def gen():
        pid = 0
        while True:
            yield env.timeout(random.expovariate(ln))
            q.put((pid, env.now, env.now))
            pid += 1
            
    def proc():
        while True:
            pid, orig_arr, att_arr = yield q.get()
            with srv.request() as req:
                yield req
                if random.random() < p:
                    destroyed = True
                else:
                    destroyed = False
                    st = random.expovariate(mu)
                    yield env.timeout(st)
                    
            if destroyed:
                q.put((pid, orig_arr, env.now))
            else:
                if env.now - att_arr > T:
                    q.put((pid, orig_arr, env.now))
                else:
                    delays.append(env.now - orig_arr)
                    
    env.process(gen())
    env.process(proc())
    env.run(until=warmup_period)
    delays.clear()
    env.run(until=warmup_period + sim_duration)
    return np.mean(delays) if delays else np.inf

def sim_destruction_parallel(ln, p):
    args_list = [(ln, p, s) for s in range(replications)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run_single, args_list))
    valid = [r for r in res if r != np.inf]
    return np.mean(valid) if valid else np.inf

def theory_destruction(ln, p):
    Lambda_star = ln / (1 - p)
    for _ in range(200):
        rho = Lambda_star * (1 - p) / mu
        if rho >= 1: return np.inf
        E_W = rho / (mu * (1 - rho))
        P_succ = (1 - p) * (1 - np.exp(-(mu - ln)*T))
        if P_succ <= 0: return np.inf
        E_A = 1 / P_succ
        Lambda_star_new = ln * E_A
        if np.isclose(Lambda_star_new, Lambda_star, atol=1e-8):
            break
        Lambda_star = Lambda_star_new
        
    return E_A * E_W + 1/mu

if __name__ == '__main__':
    print('=== Pure Destruction: Pre-Service Drop (Zero Server Time Consumed) ===')
    for p in [0.2, 0.5, 0.8]:
        for ln in [1.0, 2.33, 3.67, 5.0]:
            th = theory_destruction(ln, p)
            sim_val = sim_destruction_parallel(ln, p)
            err = abs(th - sim_val) / th * 100
            print(f'p={p:.1f}, ln={ln:.2f} | Theory: {th:.5f}, Sim: {sim_val:.5f}, Rel Err: {err:.2f}%')
