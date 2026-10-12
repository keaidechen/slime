"""Recompute environment and edge/cloud teaching budgets using only stdlib.

All timings and capacities are assumptions, not measurements. This does not
start a sandbox, call a model, exercise a network, or predict tail latency.
"""

import json
from math import isclose


def environment_budgets():
    arrival = 3.0  # tasks/s
    rounds, model_s, tool_s, tool_cores = 4, 5.0, 0.5, 2
    memory_gib = 1.5
    create_s, create_cpu_s = 0.8, 0.4
    release_s, release_cpu_s = 0.1, 0.1
    save_s, restore_s = 0.15, 0.2
    save_cpu_s = restore_cpu_s = 0.1
    base_s = rounds * (model_s + tool_s)
    tool_cpu_s = rounds * tool_s * tool_cores
    gaps = rounds - 1
    # Every transfer is a full 1.5 GiB state image; backing workspace I/O is
    # included in the common tool baseline, not in this additional snapshot I/O.
    snapshot_gib = 1.5
    policies = {
        "warm_from_start": {
            "task_s": base_s + release_s,
            "resident_s": base_s + release_s,
            "cpu_s": tool_cpu_s + release_cpu_s,
            "creates": 0,
            "restores": 0,
            "snapshot_io_gib": 0,
            "idle_slots": 6,
        },
        "create_each_turn": {
            "task_s": base_s + rounds * (create_s + release_s),
            "resident_s": rounds * (create_s + tool_s + release_s),
            "cpu_s": tool_cpu_s + rounds * (create_cpu_s + release_cpu_s),
            "creates": rounds,
            "restores": 0,
            "snapshot_io_gib": 0,
            "idle_slots": 0,
        },
        "snapshot_between_turns": {
            "task_s": base_s + create_s + gaps * (save_s + release_s + restore_s) + release_s,
            "resident_s": create_s + rounds * tool_s + gaps * (save_s + release_s + restore_s) + release_s,
            "cpu_s": tool_cpu_s + create_cpu_s + gaps * (save_cpu_s + release_cpu_s + restore_cpu_s) + release_cpu_s,
            "creates": 1,
            "restores": gaps,
            "snapshot_io_gib": gaps * 2 * snapshot_gib,
            "idle_slots": 0,
        },
        "create_once_at_first_tool": {
            "task_s": base_s + create_s + release_s,
            "resident_s": base_s - model_s + create_s + release_s,
            "cpu_s": tool_cpu_s + create_cpu_s + release_cpu_s,
            "creates": 1,
            "restores": 0,
            "snapshot_io_gib": 0,
            "idle_slots": 0,
        },
    }
    for policy in policies.values():
        policy["memory_gib_s"] = policy["resident_s"] * memory_gib
        policy["mean_resident_slots"] = arrival * policy["resident_s"] + policy["idle_slots"]
        policy["mean_memory_gib"] = policy["mean_resident_slots"] * memory_gib
        policy["mean_busy_cores"] = arrival * policy["cpu_s"]
        policy["creates_per_s"] = arrival * policy["creates"]
        policy["restores_per_s"] = arrival * policy["restores"]
        policy["snapshot_io_gib_per_s"] = arrival * policy["snapshot_io_gib"]
        policy["mean_snapshot_storage_gib"] = (
            arrival * policy["restores"] * (save_s + release_s + model_s + restore_s) * snapshot_gib
        )
        policy["mean_budget_fits"] = (
            policy["mean_memory_gib"] <= 96
            and policy["mean_busy_cores"] <= 24
            and policy["creates_per_s"] <= 5
            and policy["restores_per_s"] <= 10
            and policy["snapshot_io_gib_per_s"] <= 32
            and policy["mean_snapshot_storage_gib"] <= 128
            and policy["task_s"] <= 25
        )
    assert [p["mean_budget_fits"] for p in policies.values()] == [False, False, True, True]
    assert isclose(policies["warm_from_start"]["mean_memory_gib"], 108.45)
    assert isclose(policies["create_once_at_first_tool"]["mean_memory_gib"], 80.55)
    assert isclose(policies["snapshot_between_turns"]["task_s"], 24.25)
    assert isclose(policies["snapshot_between_turns"]["mean_memory_gib"], 19.125)
    assert isclose(policies["snapshot_between_turns"]["mean_busy_cores"], 16.2)
    # Under a 64 GiB memory budget only the snapshot candidate remains feasible.
    assert policies["snapshot_between_turns"]["mean_memory_gib"] < 64
    assert policies["create_once_at_first_tool"]["mean_memory_gib"] > 64
    # Tightening both memory and deadline leaves no feasible candidate.
    assert not any(p["mean_budget_fits"] and p["mean_memory_gib"] <= 64 and p["task_s"] <= 24
                   for p in policies.values())
    # A 10 GiB/s snapshot path cannot sustain the candidate at 3 tasks/s.
    assert 10 / policies["snapshot_between_turns"]["snapshot_io_gib"] < arrival
    assert isclose(policies["snapshot_between_turns"]["mean_snapshot_storage_gib"], 73.575)
    policies["create_once_at_first_tool"]["memory_only_tasks_per_s"] = 96 / (17.9 * memory_gib)
    return policies


def placement_budgets():
    # Decimal MB and Mbit/s: 8 * MB / Mbit_per_s gives seconds.
    upload_mb, first_chunk_mb = 0.12, 0.004
    local_s = 0.03 + 0.02 + 0.8 + 0.01 + 0.04
    paths = {
        "edge": {"up_mbps": 20, "down_mbps": 40, "rtt_s": 0.025, "queue_s": 0.02, "model_s": 0.42},
        "cloud": {"up_mbps": 8, "down_mbps": 20, "rtt_s": 0.12, "queue_s": 0.06, "model_s": 0.28},
    }
    for path in paths.values():
        stages = {
            "client_prepare": 0.03,
            "upload": 8 * upload_mb / path["up_mbps"],
            "propagation_round_trip": path["rtt_s"],
            "protocol": 0.005,
            "queue": path["queue_s"],
            "server_preprocess": 0.02,
            "model_first_chunk": path["model_s"],
            "download_first_chunk": 8 * first_chunk_mb / path["down_mbps"],
            "client_decode": 0.01,
            "initial_buffer": 0.04,
        }
        path["stages_s"] = stages
        path["first_play_s"] = sum(stages.values())
        path["fixed_s"] = path["first_play_s"] - stages["upload"]
        path["up_mbps_to_beat_local"] = 8 * upload_mb / (local_s - path["fixed_s"])
        path["weak_up_1mbps_s"] = path["fixed_s"] + 8 * upload_mb
        path["cold_first_play_s"] = path["first_play_s"] + 2 * path["rtt_s"]
        # Eight dependent rounds: 0.8 s playback per round, 0.3 s observation
        # between rounds. Assume subsequent chunks arrive without playback gaps.
        path["eight_round_s"] = 8 * (path["first_play_s"] + 0.8) + 7 * 0.3
    edge, cloud = paths["edge"], paths["cloud"]
    cloud_flip_mbps = 8 * upload_mb / (edge["first_play_s"] - cloud["fixed_s"])
    cloud_at_20 = cloud["fixed_s"] + 8 * upload_mb / 20
    assert isclose(edge["first_play_s"], 0.6188)
    assert isclose(cloud["first_play_s"], 0.6866)
    assert cloud_at_20 < edge["first_play_s"] < cloud["first_play_s"] < local_s
    assert local_s < edge["weak_up_1mbps_s"] and local_s < cloud["weak_up_1mbps_s"]
    assert cloud["cold_first_play_s"] > local_s
    # Same average bandwidth does not imply the same total upload time.
    variable_upload_s = 4 * 8 * upload_mb / 4 + 4 * 8 * upload_mb / 12
    constant_upload_s = 8 * 8 * upload_mb / 8
    assert isclose(variable_upload_s, 1.28) and isclose(constant_upload_s, 0.96)
    feature_bytes = 256 * 2048 * 2
    upload_image_and_encode_s = 8 * 0.25 / 8 + 0.01
    encode_and_upload_features_s = 0.03 + feature_bytes * 8 / 8e6
    assert feature_bytes == 1048576
    assert isclose(encode_and_upload_features_s - upload_image_and_encode_s, 0.818576)
    return {
        "local_first_play_s": local_s,
        "paths": paths,
        "cloud_up_mbps_to_beat_fixed_edge": cloud_flip_mbps,
        "cloud_first_play_at_20mbps_s": cloud_at_20,
        "eight_round_variable_upload_s": variable_upload_s,
        "eight_round_constant_upload_s": constant_upload_s,
        "cloud_connection_reuse_saving_s": 7 * 2 * cloud["rtt_s"],
        "upload_image_and_encode_s": upload_image_and_encode_s,
        "encode_and_upload_features_s": encode_and_upload_features_s,
        "fallback_after_400ms_s": 0.4 + local_s,
    }


if __name__ == "__main__":
    print(json.dumps({"environment": environment_budgets(), "placement": placement_budgets()},
                     ensure_ascii=False, indent=2))
    print("PASS assumed environment capacity and edge/cloud latency budgets; no runtime benchmark")
