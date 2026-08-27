"""
SimPy Discrete-Event Simulation Engine for Adversarial Queueing Networks.
"""

from typing import List, Optional
import random
import numpy as np
import simpy


class Packet:
    """Represents a discrete network packet tracking sojourn milestones."""

    def __init__(self, identifier: str, arrival_time: float, entry_node: int = 0):
        self.identifier = identifier
        self.original_arrival_time = arrival_time
        self.attempt_start_time = arrival_time
        self.entry_node = entry_node
        self.current_node = entry_node
        self.attempts = 1


def simulate_one_node_destruction(
    lambda_n: float,
    p: float,
    mu: float = 10.0,
    T: float = 2.0,
    backoff: float = 0.03,
    sim_duration: float = 5000.0,
    warmup_period: float = 500.0,
    seed: Optional[int] = None,
) -> float:
    """Runs a single replication of Case 1 (Pre-service destruction)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    queue = simpy.Store(env)
    delays: List[float] = []

    def packet_generator():
        pid = 0
        while True:
            yield env.timeout(random.expovariate(lambda_n))
            pkt = Packet(f"P_{pid}", env.now)
            yield queue.put(pkt)
            pid += 1

    def server_process():
        while True:
            pkt = yield queue.get()
            with server.request() as req:
                yield req
                # Pre-service destruction check
                if random.random() < p:
                    # Packet destroyed before service
                    if backoff > 0:
                        yield env.timeout(backoff)
                    pkt.attempt_start_time = env.now
                    pkt.attempts += 1
                    yield queue.put(pkt)
                    continue

                # Service time
                service_time = random.expovariate(mu)
                yield env.timeout(service_time)

                # Timeout check
                elapsed = env.now - pkt.attempt_start_time
                if elapsed > T:
                    if backoff > 0:
                        yield env.timeout(backoff)
                    pkt.attempt_start_time = env.now
                    pkt.attempts += 1
                    yield queue.put(pkt)
                else:
                    # Successful delivery
                    if env.now >= warmup_period:
                        delays.append(env.now - pkt.original_arrival_time)

    env.process(packet_generator())
    env.process(server_process())
    env.run(until=warmup_period + sim_duration)

    return float(np.mean(delays)) if delays else float(np.inf)


def simulate_one_node_modification(
    lambda_n: float,
    p: float,
    mu: float = 10.0,
    T: float = 2.0,
    sim_duration: float = 5000.0,
    warmup_period: float = 500.0,
    seed: Optional[int] = None,
) -> float:
    """Runs a single replication of Case 2 (Post-service modification)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    queue = simpy.Store(env)
    delays: List[float] = []

    def packet_generator():
        pid = 0
        while True:
            yield env.timeout(random.expovariate(lambda_n))
            pkt = Packet(f"P_{pid}", env.now)
            yield queue.put(pkt)
            pid += 1

    def server_process():
        while True:
            pkt = yield queue.get()
            with server.request() as req:
                yield req
                service_time = random.expovariate(mu)
                yield env.timeout(service_time)

                elapsed = env.now - pkt.attempt_start_time
                # Post-service attack & timeout check
                if random.random() < p or elapsed > T:
                    pkt.attempt_start_time = env.now
                    pkt.attempts += 1
                    yield queue.put(pkt)
                else:
                    if env.now >= warmup_period:
                        delays.append(env.now - pkt.original_arrival_time)

    env.process(packet_generator())
    env.process(server_process())
    env.run(until=warmup_period + sim_duration)

    return float(np.mean(delays)) if delays else float(np.inf)


def simulate_tandem(
    p: float,
    N: int = 3,
    mu: float = 2.0,
    lambda_arrival: float = 0.15,
    W: float = 8.0,
    sim_duration: float = 5000.0,
    warmup_period: float = 1000.0,
    seed: Optional[int] = None,
) -> float:
    """Runs a single replication of Case 3 (Tandem Chain)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    queues = [simpy.Store(env) for _ in range(N)]
    servers = [simpy.Resource(env, capacity=1) for _ in range(N)]
    delays: List[float] = []

    def packet_generator():
        pid = 0
        while True:
            yield env.timeout(random.expovariate(lambda_arrival))
            pkt = Packet(f"P_{pid}", env.now)
            yield queues[0].put(pkt)
            pid += 1

    def server_process(node_id: int):
        while True:
            pkt = yield queues[node_id].get()
            with servers[node_id].request() as req:
                yield req
                if node_id == 0:
                    pkt.attempt_start_time = env.now

                yield env.timeout(random.expovariate(mu))

                # Check attack or timeout
                if random.random() < p or (env.now - pkt.attempt_start_time) > W:
                    # Restart from node 0
                    yield queues[0].put(pkt)
                else:
                    if node_id < N - 1:
                        yield queues[node_id + 1].put(pkt)
                    else:
                        if env.now >= warmup_period:
                            delays.append(env.now - pkt.original_arrival_time)

    env.process(packet_generator())
    for i in range(N):
        env.process(server_process(i))

    env.run(until=warmup_period + sim_duration)
    return float(np.mean(delays)) if delays else float(np.inf)


def simulate_feedforward(
    N: int = 3,
    mu: float = 2.0,
    lambda_arr: float = 0.15,
    p: float = 0.1,
    W: float = 8.0,
    sim_duration: float = 5000.0,
    warmup_period: float = 1000.0,
    seed: Optional[int] = None,
) -> float:
    """Runs a single replication of Case 4 (Feedforward network)."""
    return simulate_tandem(
        p=p,
        N=N,
        mu=mu,
        lambda_arrival=lambda_arr,
        W=W,
        sim_duration=sim_duration,
        warmup_period=warmup_period,
        seed=seed,
    )


def simulate_feedback(
    N: int = 5,
    mu: float = 1.0,
    lambda_arr: float = 0.05,
    p: float = 0.1,
    W: float = 50.0,
    sim_duration: float = 5000.0,
    warmup_period: float = 1000.0,
    seed: Optional[int] = None,
) -> float:
    """Runs a single replication of Case 5 (Symmetric feedback mesh)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    servers = [simpy.Resource(env, capacity=1) for _ in range(N)]
    queues = [simpy.Store(env) for _ in range(N)]
    delays: List[float] = []

    def source(node_id: int):
        pid = 0
        while True:
            yield env.timeout(random.expovariate(lambda_arr))
            pkt = Packet(f"pkt_{node_id}_{pid}", env.now, entry_node=node_id)
            yield queues[node_id].put(pkt)
            pid += 1

    def server_proc(node_id: int):
        while True:
            pkt = yield queues[node_id].get()
            with servers[node_id].request() as req:
                yield req
                service_time = random.expovariate(mu)
                yield env.timeout(service_time)

                elapsed_attempt = env.now - pkt.attempt_start_time

                # Check attack or timeout
                if random.random() < p or elapsed_attempt > W:
                    # Retransmit from entry node
                    pkt.attempt_start_time = env.now
                    pkt.current_node = pkt.entry_node
                    yield queues[pkt.entry_node].put(pkt)
                else:
                    # Routing: exit with prob 1/(N+1), or route to any other node with equal prob
                    route_choices = list(range(N)) + [-1]  # -1 is exit
                    next_hop = random.choice(route_choices)
                    if next_hop == -1:
                        if env.now >= warmup_period:
                            delays.append(env.now - pkt.original_arrival_time)
                    else:
                        pkt.current_node = next_hop
                        yield queues[next_hop].put(pkt)

    for i in range(N):
        env.process(source(i))
        env.process(server_proc(i))

    env.run(until=warmup_period + sim_duration)
    return float(np.mean(delays)) if delays else float(np.inf)
