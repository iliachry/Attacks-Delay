import numpy as np
import simpy
import random
from concurrent.futures import ProcessPoolExecutor

mu = 10.0
T = 2.0
normal_traffic_rates = np.linspace(1, 5, 10)
attack_effectiveness_values = [0.2, 0.5, 0.8]
replications = 50
sim_duration = 20000
warmup_period = 2000

class Packet:
    def __init__(self, identifier, arrival_time):
        self.identifier = identifier
        self.original_arrival_time = arrival_time
        self.attempt_start_time = arrival_time

def run_sim_case(args):
    ln, p, seed, model_mode = args
    random.seed(seed)
    np.random.seed(seed)
    delays = []
    env = simpy.Environment()
    q = simpy.Store(env)
    srv = simpy.Resource(env, capacity=1)
    
    def generator():
        pid = 0
        while True:
            yield env.timeout(random.expovariate(ln))
            pkt = Packet(f"P_{pid}", env.now)
            q.put(pkt)
            pid += 1
            
    def server_proc():
        while True:
            pkt = yield q.get()
            
            with srv.request() as req:
                yield req
                
                if random.random() < p:
                    # Pre-service destruction: dropped before service
                    destroyed = True
                else:
                    destroyed = False
                    st = random.expovariate(mu)
                    yield env.timeout(st)
                    
            if destroyed:
                # Failed attempt: retransmit to queue
                pkt.attempt_start_time = env.now
                q.put(pkt)
            else:
                if (env.now - pkt.attempt_start_time) > T:
                    pkt.attempt_start_time = env.now
                    q.put(pkt)
                else:
                    delays.append(env.now - pkt.original_arrival_time)
                    
    env.process(generator())
    env.process(server_proc())
    env.run(until=warmup_period)
    delays.clear()
    env.run(until=warmup_period + sim_duration)
    return np.mean(delays) if delays else np.inf

if __name__ == '__main__':
    print("Testing clean simulation...")
