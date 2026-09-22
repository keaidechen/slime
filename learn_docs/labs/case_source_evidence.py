"""Capture/check source evidence for the four cross-system case studies.

This reads source and Git metadata without importing training frameworks.
It does not run a model or prove runtime correctness.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "train.py": ["train"],
    "slime/ray/rollout.py": ["_post_process_rewards", "_convert_samples_to_train_data"],
    "slime/observability/train_data_utils.py": ["_build_dump_payload", "save_debug_train_data"],
    "slime/observability/rollout_data_utils.py": ["save_debug_rollout_data", "load_debug_rollout_data"],
    "slime/rollout/sglang_rollout.py": ["generate_and_rm_group"],
    "slime/utils/ppo_utils.py": ["compute_policy_loss"],
    "slime/backends/megatron_utils/data.py": ["get_batch"],
    "slime/observability/profile_utils.py": ["_create_torch_profiler", "step"],
    "slime/backends/megatron_utils/actor.py": ["train_actor", "save_model", "update_weights"],
    "slime/backends/megatron_utils/loss.py": ["get_log_probs_and_entropy", "policy_loss_function", "loss_function"],
    "slime/backends/megatron_utils/cp_utils.py": ["get_sum_of_sample_mean"],
    "slime/backends/megatron_utils/model.py": ["train_one_step", "train", "save"],
    "slime/backends/megatron_utils/checkpoint.py": ["load_checkpoint", "_load_checkpoint_hf"],
    "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py": ["update_weights", "_send_weights"],
    "slime/backends/sglang_utils/sglang_engine.py": ["pause_generation", "flush_cache", "continue_generation"],
    "slime/rollout/data_source.py": ["save", "load"],
    "sglang/python/sglang/srt/managers/tokenizer_manager.py": ["generate_request", "pause_generation"],
    "sglang/python/sglang/srt/managers/scheduler.py": ["event_loop_normal", "event_loop_overlap", "abort_request", "pause_generation", "flush_cache"],
    "sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py": ["process_batch_result_prefill", "process_batch_result_decode"],
    "sglang/python/sglang/srt/managers/io_struct.py": ["PauseGenerationReqInput"],
    "sglang/python/sglang/srt/mem_cache/common.py": ["release_kv_cache"],
    "Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py": ["register_grad_ready", "start_grad_sync", "finish_grad_sync"],
    "Megatron-LM/megatron/core/distributed/finalize_model_grads.py": ["finalize_model_grads"],
    "Megatron-LM/megatron/training/checkpointing.py": ["save_checkpoint"],
}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(ROOT / repo), *args])


def definitions(node, prefix=""):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified = prefix + child.name
            yield child, qualified
            yield from definitions(child, qualified + ".")
        else:
            yield from definitions(child, prefix)


def capture():
    repos = {repo: git(repo, "rev-parse", "HEAD").decode().strip()
             for repo in (".", "sglang", "Megatron-LM")}
    files = []
    for path, wanted in SOURCES.items():
        data = (ROOT / path).read_bytes()
        text = data.decode()
        lines = text.splitlines(keepends=True)
        symbols, found = [], set()
        for node, qualified in definitions(ast.parse(text)):
            if node.name in wanted:
                found.add(node.name)
                body = "".join(lines[node.lineno - 1:node.end_lineno]).encode()
                symbols.append({"name": qualified, "line": node.lineno,
                                "end_line": node.end_lineno,
                                "sha256": hashlib.sha256(body).hexdigest()})
        if found != set(wanted):
            raise ValueError(f"Missing source symbols in {path}: {set(wanted) - found}")
        repo = next((r for r in ("sglang", "Megatron-LM") if path.startswith(r + "/")), ".")
        relative = path if repo == "." else path[len(repo) + 1:]
        head_bytes = git(repo, "show", f"HEAD:{relative}")
        files.append({"path": path, "repo": repo,
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "matches_head_blob": data == head_bytes, "symbols": symbols})
    return {"format_version": 1,
            "evidence_kind": "source-inspection-only",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "repositories": repos, "files": files}


def main():
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required to parse the inspected source files.")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path, help="Write a new source snapshot")
    group.add_argument("--check", type=Path, help="Check source against an existing snapshot")
    args = parser.parse_args()
    current = capture()
    if args.check:
        previous = json.loads(args.check.read_text())
        changed = [key for key in ("format_version", "evidence_kind", "repositories", "files")
                   if current[key] != previous.get(key)]
        if changed:
            raise SystemExit("Source evidence changed: " + ", ".join(changed))
        print(f"PASS source snapshot: {len(current['files'])} files; no runtime claim")
    elif args.output:
        args.output.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")
        print(f"Wrote {len(current['files'])} source records to {args.output}")
    else:
        print(json.dumps(current, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
