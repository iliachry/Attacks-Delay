"""
SimPy Discrete-Event Simulation Engine for Adversarial Queueing Networks.
"""
from __future__ import annotations

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
    seed: int | None = None,
) -> float:
    """Runs a single replication of Case 1 (Pre-service destruction)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    queue = simpy.Store(env)
    delays: list[float] = []

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
    seed: int | None = None,
) -> float:
    """Runs a single replication of Case 2 (Post-service modification)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    queue = simpy.Store(env)
    delays: list[float] = []

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
    seed: int | None = None,
) -> float:
    """Runs a single replication of Case 3 (Tandem Chain)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    queues = [simpy.Store(env) for _ in range(N)]
    servers = [simpy.Resource(env, capacity=1) for _ in range(N)]
    delays: list[float] = []

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
    seed: int | None = None,
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
    seed: int | None = None,
) -> float:
    """Runs a single replication of Case 5 (Symmetric feedback mesh)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    servers = [simpy.Resource(env, capacity=1) for _ in range(N)]
    queues = [simpy.Store(env) for _ in range(N)]
    delays: list[float] = []

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


def _sample_gamma_service(rng: np.random.Generator, mean_s: float, cv_sq: float) -> float:
    """Sample a positive service time with exact mean `mean_s` and squared CV `cv_sq`."""
    if mean_s <= 0.0:
        return 0.0
    if cv_sq <= 1e-9:
        return mean_s
    shape = 1.0 / cv_sq
    scale = mean_s * cv_sq
    return float(rng.gamma(shape, scale))


def simulate_cascade(
    tau: float = 0.55,
    b: int = 4,
    T_ms: float = 25.0,
    params: Any | None = None,
    q_override: float | None = None,
    sim_duration_s: float = 30.0,
    warmup_s: float = 3.0,
    seed: int = 42,
) -> dict[str, float]:
    """
    Discrete-event SimPy simulation of the two-station threshold-routed confidence cascade:
    Station 1 (M/G/1 Edge NPU) -> Threshold Router q(tau) -> Station 2 ((b, T) Batch Accumulator + M/G/1 GPU).

    Returns empirical means (in ms) for cross-validation against `aqnet.models.cascade.solve_cascade_delay`.
    """
    from aqnet.models.cascade import CascadeParams, offload_rate_q

    p = params or CascadeParams()
    q = float(q_override) if q_override is not None else offload_rate_q(tau, p)
    q = float(np.clip(q, 0.0, 1.0))

    rng = np.random.default_rng(seed)
    env = simpy.Environment()

    npu_server = simpy.Resource(env, capacity=1)
    gpu_server = simpy.Resource(env, capacity=1)
    batch_arrival_store = simpy.Store(env)

    e2e_delays_ms: list[float] = []
    d1_delays_ms: list[float] = []
    w_batch_ms: list[float] = []
    w_queue_ms: list[float] = []
    batch_sizes: list[int] = []

    T_s = T_ms / 1000.0
    E_S1_s = p.E_S1_ms / 1000.0

    def handle_frame(arrival_t: float):
        # Station 1: Edge NPU M/G/1
        with npu_server.request() as req:
            yield req
            s1 = _sample_gamma_service(rng, E_S1_s, p.C_S1_sq)
            yield env.timeout(s1)

        d1_done_t = env.now
        d1_ms = (d1_done_t - arrival_t) * 1000.0

        # Confidence threshold routing: offload with probability q
        if rng.random() >= q:
            if d1_done_t >= warmup_s:
                d1_delays_ms.append(d1_ms)
                e2e_delays_ms.append(d1_ms)
        else:
            if d1_done_t >= warmup_s:
                d1_delays_ms.append(d1_ms)
            yield batch_arrival_store.put((arrival_t, d1_done_t))

    def frame_generator():
        while True:
            ia = float(rng.exponential(1.0 / p.lam))
            yield env.timeout(ia)
            env.process(handle_frame(env.now))

    def execute_dispatched_batch(batch_items: list[tuple[float, float]], dispatch_t: float):
        B_cur = len(batch_items)
        if dispatch_t >= warmup_s:
            batch_sizes.append(B_cur)
            for _, t_enter_batch in batch_items:
                w_batch_ms.append((dispatch_t - t_enter_batch) * 1000.0)

        # Network serialization & DMA transfer time T_net(B_cur)
        t_net_s = (p.t_net_0_ms + p.t_net_1_ms * B_cur) / 1000.0
        # Station 2 GPU M/G/1 queue
        q_enter_t = env.now
        with gpu_server.request() as req:
            yield req
            q_start_t = env.now
            t_gpu_mean_s = (p.t_gpu_0_ms + p.t_gpu_1_ms * B_cur) / 1000.0
            s2_gpu = _sample_gamma_service(rng, t_gpu_mean_s, p.C_S2_sq)
            yield env.timeout(s2_gpu)

        # Network pipeline + GPU completion
        yield env.timeout(t_net_s)
        done_t = env.now

        if done_t >= warmup_s:
            wq_ms = (q_start_t - q_enter_t) * 1000.0
            for orig_t, _ in batch_items:
                w_queue_ms.append(wq_ms)
                e2e_delays_ms.append((done_t - orig_t) * 1000.0)

    def batch_accumulator():
        while True:
            # Wait for the first offloaded frame to start a renewal batch cycle
            first_item = yield batch_arrival_store.get()
            batch_items = [first_item]
            if b == 1 or T_s <= 0.0:
                env.process(execute_dispatched_batch(batch_items, env.now))
                continue

            deadline = env.now + T_s
            while len(batch_items) < b:
                rem = deadline - env.now
                if rem <= 1e-12:
                    break
                get_ev = batch_arrival_store.get()
                timeout_ev = env.timeout(rem)
                res = yield simpy.AnyOf(env, [get_ev, timeout_ev])
                if get_ev in res.events:
                    batch_items.append(get_ev.value)
                else:
                    # Timeout expired; cancel the pending get request
                    if get_ev.triggered:
                        batch_items.append(get_ev.value)
                    else:
                        batch_arrival_store.get_queue.remove(get_ev)
                    break

            env.process(execute_dispatched_batch(batch_items, env.now))

    env.process(frame_generator())
    env.process(batch_accumulator())
    env.run(until=warmup_s + sim_duration_s)

    return {
        "E_D_ms": float(np.mean(e2e_delays_ms)) if e2e_delays_ms else float("inf"),
        "E_D1_ms": float(np.mean(d1_delays_ms)) if d1_delays_ms else float("inf"),
        "E_W_batch_ms": float(np.mean(w_batch_ms)) if w_batch_ms else 0.0,
        "E_W_queue_ms": float(np.mean(w_queue_ms)) if w_queue_ms else 0.0,
        "E_B_disp": float(np.mean(batch_sizes)) if batch_sizes else 1.0,
        "n_frames": len(e2e_delays_ms),
    }
