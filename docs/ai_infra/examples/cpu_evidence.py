"""Run real local CPU measurements and interrupted-process recovery experiments.

Only Python's standard library is required. Creates a fresh output directory.
No GPU, distributed backend, or power-loss durability claim is made.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from mini_transformer import forward, gradients, initial_parameters


SAMPLES = [([0, 1], [1, 2]), ([2], [0])]
DATA_HASH = hashlib.sha256(json.dumps(SAMPLES).encode()).hexdigest()


def initial_state():
    parameters = initial_parameters()
    zeros = {name: [[0.0 for _ in row] for row in matrix] for name, matrix in parameters.items()}
    return {"parameters": parameters, "m": zeros, "v": copy.deepcopy(zeros),
            "step": 0, "cursor": 0, "history": [], "data_hash": DATA_HASH,
            "config": {"lr": 0.01, "beta1": 0.9, "beta2": 0.99, "epsilon": 1e-8}}


def train_step(state):
    sample_id = state["cursor"] % len(SAMPLES)
    tokens, labels = SAMPLES[sample_id]
    loss, grads, _ = gradients(state["parameters"], tokens, labels)
    state["step"] += 1
    step, config = state["step"], state["config"]
    for name, matrix in state["parameters"].items():
        for i, row in enumerate(matrix):
            for j in range(len(row)):
                gradient = grads[name][i][j]
                m = config["beta1"] * state["m"][name][i][j] + (1-config["beta1"])*gradient
                v = config["beta2"] * state["v"][name][i][j] + (1-config["beta2"])*gradient**2
                state["m"][name][i][j], state["v"][name][i][j] = m, v
                row[j] -= config["lr"] * (m/(1-config["beta1"]**step)) / (
                    math.sqrt(v/(1-config["beta2"]**step)) + config["epsilon"])
    state["cursor"] += 1
    state["history"].append({"sample_id": sample_id, "loss": loss})


def worker(directory, crash):
    checkpoint = directory / "checkpoint.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else initial_state()
    if state["data_hash"] != DATA_HASH or state["config"] != initial_state()["config"]:
        raise ValueError("Checkpoint data or optimizer recipe mismatch")
    while state["step"] < 6:
        start = time.perf_counter_ns()
        train_step(state)
        duration = time.perf_counter_ns() - start
        with (directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"name": "forward_backward_adam", "ph": "X",
                                    "ts": start/1000, "dur": duration/1000,
                                    "pid": os.getpid(), "tid": 0,
                                    "args": {"step": state["step"]}}) + "\n")
        staged = directory / "checkpoint.staged.json"
        with staged.open("w") as stream:
            json.dump(state, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        if state["step"] == 3 and crash == "before":
            os._exit(23)
        os.replace(staged, checkpoint)
        if state["step"] == 3 and crash == "after":
            os._exit(23)
    (directory / "final.json").write_text(json.dumps(state, sort_keys=True))


def global_gradient_check():
    parameters = initial_parameters()
    local = [gradients(parameters, *sample)[1] for sample in SAMPLES]
    maximum = 0.0
    wrong_maximum = 0.0
    for name, matrix in parameters.items():
        for i, row in enumerate(matrix):
            for j, original in enumerate(row):
                row[j] = original + 1e-5
                plus = sum(len(s[0])*forward(parameters, *s)[0].data for s in SAMPLES) / 3
                row[j] = original - 1e-5
                minus = sum(len(s[0])*forward(parameters, *s)[0].data for s in SAMPLES) / 3
                row[j] = original
                expected = (plus-minus)/2e-5
                combined = (2*local[0][name][i][j]+local[1][name][i][j])/3
                wrong = (local[0][name][i][j]+local[1][name][i][j])/2
                maximum = max(maximum, abs(combined-expected))
                wrong_maximum = max(wrong_maximum, abs(wrong-expected))
    assert maximum < 2e-6 and wrong_maximum > 1e-3
    return {"weighted_gradient_max_error": maximum, "unweighted_negative_control_error": wrong_maximum}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    events = []
    rows = [{"id": i, "tokens": [(i+j) % 97 for j in range(32)]} for i in range(512)]
    fixture = output / "fixture.jsonl"
    fixture.write_text("".join(json.dumps(row)+"\n" for row in rows))
    expected = sum(sum(row["tokens"]) for row in rows)
    raw_times = {"parse_each_pass": [], "reuse_decoded": []}
    load_start = time.perf_counter_ns()
    decoded = [json.loads(line) for line in fixture.read_text().splitlines()]
    setup_ms = (time.perf_counter_ns()-load_start)/1e6
    # Warm both paths; alternation reduces but does not remove order effects.
    for repeat in range(8):
        names = list(raw_times) if repeat % 2 == 0 else list(reversed(raw_times))
        for name in names:
            start = time.perf_counter_ns()
            if name == "parse_each_pass":
                with fixture.open() as stream:
                    checksum = sum(sum(json.loads(line)["tokens"]) for line in stream)
            else:
                checksum = sum(sum(row["tokens"]) for row in decoded)
            duration = time.perf_counter_ns()-start
            assert checksum == expected
            events.append({"name": name, "ph": "X", "ts": start/1000, "dur": duration/1000,
                           "pid": os.getpid(), "tid": 0, "args": {"repeat": repeat}})
            if repeat:
                raw_times[name].append(duration/1e6)

    checks = global_gradient_check()
    executable = [sys.executable, str(Path(__file__).resolve())]
    baseline = output / "continuous"
    baseline.mkdir()
    subprocess.run(executable + ["--worker", str(baseline)], check=True)
    expected_state = json.loads((baseline / "final.json").read_text())
    recovery = {}
    for crash in ("before", "after"):
        folder = output / crash
        folder.mkdir()
        result = subprocess.run(executable + ["--worker", str(folder), "--crash", crash], check=False)
        assert result.returncode == 23
        committed = json.loads((folder / "checkpoint.json").read_text())
        assert committed["step"] == (2 if crash == "before" else 3)
        if crash == "before":
            broken = output / "optimizer_reset"
            broken.mkdir()
            wrong = copy.deepcopy(committed)
            wrong["m"] = initial_state()["m"]
            wrong["v"] = initial_state()["v"]
            (broken / "checkpoint.json").write_text(json.dumps(wrong))
        subprocess.run(executable + ["--worker", str(folder)], check=True)
        resumed = json.loads((folder / "final.json").read_text())
        assert resumed == expected_state
        recovery[crash] = {"exit_code": result.returncode, "resumed_from_step": committed["step"],
                           "parameters_optimizer_cursor_history_exact_match": True}
    subprocess.run(executable + ["--worker", str(broken)], check=True)
    wrong_state = json.loads((broken / "final.json").read_text())
    assert wrong_state["parameters"] != expected_state["parameters"]
    for path in output.glob("*/events.jsonl"):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    origin = min(event["ts"] for event in events)
    for event in events:
        event["ts"] -= origin
    trace = {"traceEvents": sorted(events, key=lambda event: event["ts"]), "displayTimeUnit": "ms"}
    (output / "trace.json").write_text(json.dumps(trace))
    report = {
        "scope": "CPU Python arithmetic, application cache, process crash; no GPU or power-loss test",
        "environment": {"python": sys.version, "platform": platform.platform()},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(__file__).with_name("mini_transformer.py"))},
        "fixture": {"rows": len(rows), "tokens_per_row": 32, "bytes": fixture.stat().st_size,
                    "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "checksum": expected},
        "cache_setup_ms": setup_ms, "raw_ms": raw_times,
        "median_ms": {name: statistics.median(values) for name, values in raw_times.items()},
        "cache_state": "OS cache uncontrolled; application cache built once; first pair excluded",
        "gradient_check": checks, "recovery": recovery,
        "optimizer_reset_negative_control_detected": True,
        "continuous_losses": [row["loss"] for row in expected_state["history"]],
        "trace_events": len(events), "artifacts": ["fixture.jsonl", "trace.json", "*/checkpoint.json", "*/final.json"],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--crash", choices=("none", "before", "after"), default="none")
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.crash)
    elif args.output is not None:
        run(args.output)
    else:
        parser.error("provide a fresh --output directory")
