"""Recompute the training and serving design cases using only the standard library.

All service times, overlap fractions and activation sizes are teaching inputs,
not measurements or predictions of a named GPU/model. Run from the repository:
    python docs/ai_infra/examples/design_cases.py
    python docs/ai_infra/examples/design_cases.py --json
"""

import argparse
import json
from dataclasses import dataclass
from math import isclose

GIB = 2**30
PARAMETERS = 8_000_000_000
WEIGHTS_GIB = 2 * PARAMETERS / GIB
KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2
GPU_COUNT = 8
GPU_GIB = 64
RESERVE_GIB = 6
TRAIN_STEPS = 4000
GLOBAL_BATCH = 128
MICRO_BATCH = 2
SEQUENCE_LENGTH = 4096


@dataclass(frozen=True)
class TrainingPlan:
    name: str
    tp: int
    dp: int
    sharding: str
    activation_gib: float
    micro_compute_s: float
    dp_overlap: float = 0.75
    tp_wire_gib: float = 0
    tp_latency_s: float = 0


TRAINING_PLANS = (
    TrainingPlan("DP8", 1, 8, "replicated", 42, 0.28),
    TrainingPlan("ZeRO2-DP8, no recompute", 1, 8, "zero2", 42, 0.28),
    TrainingPlan("ZeRO2-DP8, recompute", 1, 8, "zero2", 14, 0.36),
    TrainingPlan("FSDP8, no recompute", 1, 8, "full", 42, 0.28, 0.65),
    TrainingPlan("FSDP8, recompute", 1, 8, "full", 14, 0.36, 0.65),
    TrainingPlan("TP2 x ZeRO2-DP4, recompute", 2, 4, "zero2", 8, 0.21, 0.75, 48, 0.04),
    TrainingPlan("TP4 x ZeRO2-DP2, no recompute", 4, 2, "zero2", 12, 0.085, 0.75, 96, 0.08),
)


def training_case(dp_bandwidth_gib_s):
    # Checkpoints save weights, FP32 master weights and two FP32 moments;
    # gradients are not needed at this optimizer-step boundary.
    checkpoint_gib = 14 * PARAMETERS / GIB
    checkpoint_s = checkpoint_gib / 8 + 2
    restore_s = checkpoint_gib / 8 + 45
    fixed_s = 20 * checkpoint_s + 60 + 120 + restore_s
    rows = []
    for plan in TRAINING_PLANS:
        assert plan.tp * plan.dp == GPU_COUNT
        accumulation = GLOBAL_BATCH // (MICRO_BATCH * plan.dp)
        assert accumulation * MICRO_BATCH * plan.dp == GLOBAL_BATCH
        weights = WEIGHTS_GIB / plan.tp
        # State bytes per parameter: BF16 weight 2, gradient 2,
        # FP32 master weight 4, Adam moments 4 + 4. Same in every candidate.
        if plan.sharding == "replicated":
            state = weights * 8
            dp_wire = 2 * (plan.dp - 1) / plan.dp * weights
            materialized = 0
        elif plan.sharding == "zero2":
            state = weights + weights * 7 / plan.dp
            # Reduce-scatter every microbatch; parameter all-gather once/update.
            dp_wire = (accumulation + 1) * (plan.dp - 1) / plan.dp * weights
            materialized = 0
        else:
            state = weights * 8 / plan.dp
            # Forward/backward all-gathers and gradient reduce-scatter every MB.
            dp_wire = 3 * accumulation * (plan.dp - 1) / plan.dp * weights
            materialized = 2  # Explicit extra live full-layer/prefetch buffers.
        peak = state + plan.activation_gib + materialized + RESERVE_GIB
        dp_exposed = dp_wire / dp_bandwidth_gib_s * (1 - plan.dp_overlap)
        tp_exposed = (plan.tp_wire_gib / 80 + plan.tp_latency_s) * 0.5
        # Includes backward/recompute in compute; optimizer 40 ms + data wait 60 ms.
        step = accumulation * plan.micro_compute_s + dp_exposed + tp_exposed + 0.10
        # One failure with expected 100 lost steps in a 200-step save interval.
        hours = ((TRAIN_STEPS + 100) * step + fixed_s) / 3600
        rows.append({
            "name": plan.name, "accumulation": accumulation,
            "peak_gib": peak, "dp_wire_gib": dp_wire,
            "step_s": step if peak <= GPU_GIB else None,
            "hours": hours if peak <= GPU_GIB else None,
            "feasible": peak <= GPU_GIB and hours <= 5,
        })
    feasible = [row for row in rows if row["feasible"]]
    return {
        "dp_bandwidth_gib_s": dp_bandwidth_gib_s,
        "tokens": TRAIN_STEPS * GLOBAL_BATCH * SEQUENCE_LENGTH,
        "checkpoint_gib": checkpoint_gib, "checkpoint_s": checkpoint_s,
        "restore_s": restore_s, "fixed_s": fixed_s,
        "rows": rows, "selected": min(feasible, key=lambda row: row["hours"])["name"],
    }


@dataclass(frozen=True)
class ServingPlan:
    name: str
    tp: int
    short_prefill_s: float
    long_prefill_s: float
    mixed_decode_s: float
    long_decode_s: float
    weight_gib: float = WEIGHTS_GIB


SERVING_PLANS = (
    ServingPlan("8 x TP1", 1, 0.050, 0.500, 0.012, 0.010),
    ServingPlan("4 x TP2", 2, 0.032, 0.280, 0.009, 0.007),
    ServingPlan("2 x TP4", 4, 0.022, 0.170, 0.010, 0.008),
    ServingPlan("1 x TP8", 8, 0.018, 0.120, 0.011, 0.009),
    ServingPlan("8 x TP1, warm prefix", 1, 0.050, 0.180, 0.012, 0.010),
    ServingPlan("8 x TP1, W8 weights", 1, 0.050, 0.500, 0.009, 0.007,
                WEIGHTS_GIB / 2 + 0.2),
)


def serving_case(period_s=5.0, ttft_limit_s=1.0):
    rows = []
    for plan in SERVING_PLANS:
        replicas = GPU_COUNT // plan.tp
        short, long = 16 // replicas, 8 // replicas
        assert short * replicas == 16 and long * replicas == 8
        prefill = short * plan.short_prefill_s + long * plan.long_prefill_s
        # Prefill supplies token 1; 255 further steps complete short requests,
        # then 256 further steps complete long requests (512 outputs in total).
        completion = prefill + 255 * plan.mixed_decode_s + 256 * plan.long_decode_s
        kv_gib = (short * (1024 + 256) + long * (8192 + 512)) * KV_BYTES_PER_TOKEN / GIB
        peak = plan.weight_gib / plan.tp + RESERVE_GIB + kv_gib / plan.tp
        feasible = (peak <= GPU_GIB and completion <= period_s
                    and prefill <= ttft_limit_s and completion <= 6.5
                    and max(plan.mixed_decode_s, plan.long_decode_s) <= 0.015)
        rows.append({
            "name": plan.name, "peak_gib": peak, "ttft_s": prefill,
            "max_itl_ms": 1000 * max(plan.mixed_decode_s, plan.long_decode_s),
            "completion_s": completion, "min_period_s": completion,
            "timing_feasible": feasible,
            "quality": "unvalidated W8" if "W8" in plan.name else "BF16 reference contract",
        })
    # Four TP1 P workers and four TP1 D workers. Each receives 4 short + 2 long.
    prefill = 4 * 0.050 + 2 * 0.500
    transfer_gib = (16 * 1024 + 8 * 8192) * KV_BYTES_PER_TOKEN / GIB
    # 100 GiB/s aggregate effective bandwidth, four balanced 25 GiB/s paths.
    transfer = transfer_gib / 100 + 0.010
    decode = 255 * 0.008 + 256 * 0.006
    completion = prefill + transfer + decode
    min_period = max(prefill + transfer, decode)
    peak = WEIGHTS_GIB + RESERVE_GIB + 2.75
    rows.append({
        "name": "4P + 4D, TP1", "peak_gib": peak,
        "ttft_s": prefill + transfer, "max_itl_ms": 8,
        "completion_s": completion, "min_period_s": min_period,
        "timing_feasible": (peak <= GPU_GIB and min_period <= period_s
                            and prefill + transfer <= ttft_limit_s and completion <= 6.5),
        "quality": "BF16 reference contract",
    })
    return {
        "period_s": period_s, "requests_per_s": 24 / period_s,
        "ttft_limit_s": ttft_limit_s, "transfer_gib": transfer_gib, "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit full precision results as JSON")
    args = parser.parse_args()
    results = {
        "model": {"weight_gib": WEIGHTS_GIB, "kv_bytes_per_token": KV_BYTES_PER_TOKEN},
        "training": training_case(80), "training_slow_dp": training_case(20),
        "serving": serving_case(), "serving_faster_arrivals": serving_case(4, 1.5),
    }
    assert KV_BYTES_PER_TOKEN == 128 * 1024
    assert isclose(results["serving"]["transfer_gib"], 10)
    assert results["training"]["selected"] == "ZeRO2-DP8, recompute"
    assert results["training_slow_dp"]["selected"] == "TP4 x ZeRO2-DP2, no recompute"
    if args.json:
        print(json.dumps(results, indent=2))
        return
    print("TEACHING INPUTS ONLY: these are not benchmark results.")
    print(f"Weights: {WEIGHTS_GIB:.3f} GiB; KV: {KV_BYTES_PER_TOKEN // 1024} KiB/token")
    for key in ("training", "training_slow_dp"):
        case = results[key]
        print(f"\nTraining, DP bandwidth {case['dp_bandwidth_gib_s']} GiB/s")
        for row in case["rows"]:
            timing = (f"{row['step_s']:.3f} s/step, {row['hours']:.3f} h"
                      if row["hours"] is not None else "does not fit")
            print(f"{row['name']}: {row['peak_gib']:.3f} GiB, {timing}, feasible={row['feasible']}")
        print(f"Selected: {case['selected']}")
    for key in ("serving", "serving_faster_arrivals"):
        case = results[key]
        print(f"\nServing, period {case['period_s']} s, TTFT limit {case['ttft_limit_s']} s")
        for row in case["rows"]:
            print(f"{row['name']}: TTFT={row['ttft_s']:.3f} s, "
                  f"finish={row['completion_s']:.3f} s, min period={row['min_period_s']:.3f} s, "
                  f"timing feasible={row['timing_feasible']}; {row['quality']}")


if __name__ == "__main__":
    main()
