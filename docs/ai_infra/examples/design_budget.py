"""Recompute teaching workload and budget examples; no engine or GPU benchmark.

Run: python3 docs/ai_infra/examples/design_budget.py
"""

from fractions import Fraction
import json
from math import isclose, sqrt


def fcfs(arrivals, service_seconds=1):
    available = 0
    responses = []
    for arrival in arrivals:
        start = max(arrival, available)
        available = start + service_seconds
        responses.append(available - arrival)
    return responses


def main():
    gib = 2**30
    kv_bytes = 2 * 32 * 8 * 128 * 2
    reservations = [Fraction((1024 + 256) * kv_bytes, gib),
                    Fraction((8192 + 512) * kv_bytes, gib)]
    lifetimes = [2, 8]
    expected_occupancy = sum(m * t for m, t in zip(reservations, lifetimes))

    # Each class arrives once per second. Integrate actual reservations across
    # event intervals after warmup, independently of the occupancy identity.
    objects = [(arrival, arrival + lifetime, size)
               for arrival in range(-8, 20)
               for size, lifetime in zip(reservations, lifetimes)]
    events = sorted({0, 20} | {time for start, end, _ in objects
                               for time in (start, end) if 0 < time < 20})
    integral = Fraction(0)
    for left, right in zip(events, events[1:]):
        resident = sum(size for start, end, size in objects if start <= left < end)
        integral += resident * (right - left)
    assert integral / 20 == expected_occupancy == Fraction(141, 16)
    naive_occupancy = sum(reservations) / 2 * sum(lifetimes)
    assert naive_occupancy == Fraction(195, 32)
    assert naive_occupancy < expected_occupancy

    uniform = fcfs([0, 1, 2, 3])
    burst = fcfs([0, 0, 0, 0])
    assert uniform == [1, 1, 1, 1] and burst == [1, 2, 3, 4]
    assert sum(t <= 2 for t in uniform) == 4
    assert sum(t <= 2 for t in burst) == 2
    new_input_before = 1024 + 8192
    new_input_after = 1024 + (8192 - 6144)
    assert new_input_before == 9216 and new_input_after == 3072

    # Normalized fit A=B=1, alpha=beta=1/2 and budget ND=4.
    # Enumerate candidate model sizes; compare with the analytic optimum N=D=2.
    losses = {n: 1 / sqrt(n) + 1 / sqrt(4 / n)
              for n in [0.25, 0.5, 1, 2, 4, 8, 16]}
    assert min(losses, key=losses.get) == 2
    assert isclose(losses[2], sqrt(2))

    fixed = [10000, 16000]
    per_attempt = [Fraction(4, 100), Fraction(27, 1000)]
    pass_rate = [Fraction(8, 10), Fraction(9, 10)]
    per_qualified = [cost / rate for cost, rate in zip(per_attempt, pass_rate)]
    crossover = (fixed[1] - fixed[0]) / (per_qualified[0] - per_qualified[1])
    assert crossover == 300000
    totals = {q: [base + cost * q for base, cost in zip(fixed, per_qualified)]
              for q in [100000, 300000, 1000000]}
    assert totals[100000] == [15000, 19000]
    assert totals[300000][0] == totals[300000][1]
    assert totals[1000000] == [60000, 46000]

    cost_per_million = [8 / (1000 * 3600 * busy / 1e6) for busy in [0.25, 0.5]]
    assert isclose(cost_per_million[0], 80 / 9)
    assert isclose(cost_per_million[0], 2 * cost_per_million[1])
    print(json.dumps({
        "evidence": "teaching arithmetic and deterministic events; no measured performance",
        "weights_gib": 8e9 * 2 / gib,
        "kv_bytes_per_token": kv_bytes,
        "mean_reserved_gib": float(expected_occupancy),
        "incorrect_independent_gib": float(naive_occupancy),
        "uniform_response_seconds": uniform,
        "burst_response_seconds": burst,
        "new_input_tokens_per_second": [new_input_before, new_input_after],
        "normalized_scaling_optimum": {"parameters": 2, "tokens": 2},
        "qualified_task_cost_crossover": int(crossover),
        "costs_by_qualified_tasks": {q: list(map(float, costs)) for q, costs in totals.items()},
        "gpu_cost_per_million_output_tokens": cost_per_million,
    }, indent=2))


if __name__ == "__main__":
    main()
