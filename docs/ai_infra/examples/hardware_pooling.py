"""Reproduce analytical examples for accelerators, remote KV and AF splitting.

All parameters are synthetic teaching inputs, not measured hardware results.
Run from the repository root: python3 docs/ai_infra/examples/hardware_pooling.py
"""

from math import ceil

GIB = 2**30
KIB = 2**10
MIB = 2**20


def accelerator_tradeoffs():
    """A projection changes from bandwidth-bound to compute-bound as M grows."""
    k, n = 8192, 4096
    large_batch_saving = None
    for rows in (1, 512):
        flops = 2 * rows * k * n
        byte_count = 2 * (rows * k + k * n + rows * n)
        time_a = 5e-6 + max(flops / 100e12, byte_count / 1e12)
        time_b = 10e-6 + max(flops / 200e12, byte_count / 500e9)
        print(
            f"projection M={rows}: FLOPs={flops:,}, bytes={byte_count:,}, "
            f"intensity={flops / byte_count:.6f}, "
            f"A={time_a * 1e6:.6f} us, B={time_b * 1e6:.6f} us"
        )
        if rows == 512:
            large_batch_saving = time_a - time_b
    print(f"900 s extra startup breaks even at {ceil(900 / large_batch_saving):,} calls")
    print(f"instance capacity requirement: {40 + 4 + 8} GiB")


def remote_kv_pool():
    """Separate capacity, per-transfer latency and sustained shared bandwidth."""
    gpu_count = 4 * 8
    local_kv_bytes = (80 - 20 - 8 - 8) * GIB
    kv_per_token = 2 * 32 * 8 * 128 * 2
    session_bytes = 32768 * kv_per_token
    sessions_per_gpu = local_kv_bytes // session_bytes
    raw_pool_bytes = (1024 - 128) * GIB
    logical_pool_bytes = raw_pool_bytes // 2
    print(
        f"KV: {kv_per_token // KIB} KiB/token, {session_bytes / GIB:g} GiB/session; "
        f"{sessions_per_gpu} sessions/GPU, {gpu_count * sessions_per_gpu} sessions/rack"
    )
    print(
        f"rack: {gpu_count * 80} GiB total, {gpu_count * local_kv_bytes / GIB:g} GiB KV; "
        f"pool: {logical_pool_bytes / GIB:g} GiB logical, "
        f"{logical_pool_bytes // session_bytes} sessions with two replicas"
    )
    print(f"80 idle sessions: {80 * session_bytes * 2 / GIB:g} GiB physical with two replicas")
    path_bandwidth, completion_latency, transaction_bytes = 25e9, 8e-6, 4 * KIB
    limited_bandwidth = min(path_bandwidth, 16 * transaction_bytes / completion_latency)
    required_inflight = ceil(path_bandwidth * completion_latency / transaction_bytes)
    print(
        f"16 inflight reads: {limited_bandwidth / 1e9:.3f} GB/s; "
        f"inflight reads needed for 25 GB/s: {required_inflight}"
    )
    print(
        f"4 GiB restore: {session_bytes / path_bandwidth * 1e3:.6f} ms at 25 GB/s; "
        f"{session_bytes / limited_bandwidth * 1e3:.6f} ms with 16 inflight reads; "
        f"local read at 1 TB/s: {session_bytes / 1e12 * 1e3:.6f} ms"
    )
    print(f"8 restores sharing 25 GB/s: drain time >= {8 * session_bytes / path_bandwidth:.6f} s")
    for restore_rate in (2, 8):
        print(f"{restore_rate} restores/s demand {restore_rate * session_bytes / 1e9:.6f} GB/s")


def attention_ffn_split():
    """Latency savings and capacity bounds use distinct resource accounting."""
    layers, batch, hidden, element_bytes = 32, 32, 4096, 2
    data_bytes = 2 * layers * batch * hidden * element_bytes
    colocated_time = layers * (90e-6 + 150e-6)
    attention_demand = layers * 75e-6
    ffn_demand = layers * 90e-6
    split_compute = attention_demand + ffn_demand
    compute_saving = colocated_time - split_compute
    print(f"AF: {data_bytes / MIB:g} MiB/decode batch across {2 * layers} handoffs")
    base_split_time = None
    for bandwidth, startup in ((25e9, 8e-6), (5e9, 8e-6), (25e9, 40e-6)):
        communication = 2 * layers * startup + data_bytes / bandwidth
        split_time = split_compute + communication
        print(
            f"AF {bandwidth / 1e9:g} GB/s, {startup * 1e6:g} us startup: "
            f"communication={communication * 1e3:.6f} ms, "
            f"total={split_time * 1e3:.6f} ms, "
            f"latency saving={(1 - split_time / colocated_time) * 100:.4f}%"
        )
        if bandwidth == 25e9 and startup == 8e-6:
            base_split_time = split_time
    bandwidth_threshold = data_bytes / (compute_saving - 2 * layers * 8e-6)
    startup_threshold = (compute_saving - data_bytes / 25e9) / (2 * layers)
    print(
        f"AF threshold: bandwidth > {bandwidth_threshold / 1e9:.6f} GB/s at 8 us; "
        f"startup < {startup_threshold * 1e6:.6f} us at 25 GB/s"
    )
    print(
        f"AF no bandwidth can compensate when "
        f"startup >= {compute_saving / (2 * layers) * 1e6:g} us"
    )
    print(
        f"8 cards without pipelining: colocated={8 * batch / colocated_time:.3f} token/s; "
        f"4 AF pairs={4 * batch / base_split_time:.3f} token/s"
    )
    stage_bounds = (
        4 * batch / attention_demand,
        4 * batch / ffn_demand,
        25e9 * batch / data_bytes,
    )
    print(
        "AF pipelined capacity bounds (not forecasts), A/F/network token/s: "
        + ", ".join(f"{x:.3f}" for x in stage_bounds)
    )
    print(f"AF optimistic capacity upper bound: {min(stage_bounds):.3f} token/s")


if __name__ == "__main__":
    print("Analytical teaching inputs only; no hardware benchmark is performed.")
    accelerator_tradeoffs()
    remote_kv_pool()
    attention_ffn_split()
