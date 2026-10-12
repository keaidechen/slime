"""Verify the Omni articles' teaching arithmetic and state transitions on CPU.

Python standard library only. No model, audio device, GPU or upstream code is run.
Use --source-root PATH to also check a local vLLM-Omni checkout against the
fixed source manifest; this checks file identity, not runtime behavior.
"""

import argparse
import hashlib
import json
from math import isclose
from pathlib import Path


def representations():
    samples = 10 * 16000
    frames = (samples - 400) // 160 + 1
    encoded_frames = (frames + 3) // 4  # Explicit teaching shape rule.
    feature_bytes = encoded_frames * 1024 * 2
    kv_per_position = 2 * 32 * 8 * 128 * 2
    assert (samples, frames, encoded_frames) == (160000, 998, 250)
    assert samples * 2 == 320000
    assert feature_bytes == 512000
    assert kv_per_position == 131072
    assert (encoded_frames + 128) * kv_per_position / 2**20 == 47.25
    assert encoded_frames * kv_per_position / feature_bytes == 64
    codec_frames, books, entries = 10 * 50, 8, 1024
    bits = entries.bit_length() - 1
    assert 2**bits == entries
    assert codec_frames * books == 4000
    assert 50 * books * bits == 4000  # Packed bits/s, no overhead.
    assert codec_frames * books * 2 == 8000  # int16 storage, bytes.
    assert 24000 * 200 // 1000 * 2 == 9600
    vector, first, second = [.7, -.2], [.5, 0], [.25, -.25]
    reconstructed = [a + b for a, b in zip(first, second)]
    squared_error = sum((x - y)**2 for x, y in zip(vector, reconstructed))
    assert isclose(squared_error, .005)
    return {"mel_frames": frames, "encoder_frames": encoded_frames,
            "feature_bytes": feature_bytes, "initial_kv_mib": 47.25,
            "codec_indices": 4000, "rvq_squared_error": squared_error}


def playback_timeline():
    assert sum([20, 40, 60, 30, 10, 20, 40]) == 220
    arrivals = [180, 340, 680, 770]
    cursor, spans, gaps = 220, [], []
    for arrival in arrivals:
        start = max(cursor, arrival)
        gaps.append(start - cursor)
        spans.append([start, start + 200])
        cursor = start + 200
    assert spans == [[220, 420], [420, 620], [680, 880], [880, 1080]]
    assert gaps == [0, 0, 60, 0]
    assert cursor - 220 == 800 + sum(gaps)
    played_at_interrupt = sum(max(0, min(500, end) - start) for start, end in spans)
    generated, sent, received = 600, 400, 400
    assert played_at_interrupt == 280
    assert received - played_at_interrupt == 120
    assert generated - played_at_interrupt == 320
    assert played_at_interrupt <= received <= sent <= generated
    assert 530 - 500 == 30  # Separate nonzero device-stop-latency example.
    assert 600 / 800 == .75  # RTF does not bound the gaps above.
    return {"first_play_ms": 220, "play_spans_ms": spans,
            "gap_ms": sum(gaps), "played_at_500_ms": played_at_interrupt}


class ChunkReceiver:
    """A bounded, generation-aware toy receiver; ready blocks are not yet played."""

    def __init__(self, epoch, capacity=4):
        self.epoch = epoch
        self.capacity = capacity
        self.next_id = 0
        self.pending = {}
        self.ready = []

    def receive(self, epoch, chunk_id, value):
        if epoch != self.epoch:
            return "stale"
        if chunk_id < self.next_id or chunk_id in self.pending:
            return "duplicate"
        if len(self.pending) + len(self.ready) >= self.capacity:
            return "backpressure"
        # Bound the reorder window so future blocks cannot fill every slot
        # while preventing the missing prefix from entering the buffer.
        if chunk_id >= self.next_id + self.capacity - len(self.ready):
            return "backpressure"
        self.pending[chunk_id] = value
        while self.next_id in self.pending:
            self.ready.append(self.pending.pop(self.next_id))
            self.next_id += 1
        return "accepted"

    def cancel(self):
        self.epoch += 1
        self.next_id = 0
        self.pending.clear()
        self.ready.clear()


def chunk_identity():
    receiver = ChunkReceiver(epoch=7)
    assert receiver.receive(7, 1, "second") == "accepted"
    assert receiver.ready == []  # Never play past a missing prefix.
    assert receiver.receive(7, 1, "duplicate") == "duplicate"
    assert receiver.receive(7, 0, "first") == "accepted"
    assert receiver.ready == ["first", "second"]
    assert receiver.receive(7, 0, "duplicate") == "duplicate"
    assert receiver.receive(7, 3, "fourth") == "accepted"
    assert receiver.receive(7, 2, "third") == "accepted"
    assert receiver.ready == ["first", "second", "third", "fourth"]
    assert receiver.receive(7, 4, "overflow") == "backpressure"
    receiver.cancel()
    assert not receiver.pending and not receiver.ready
    assert receiver.receive(7, 4, "late-old-output") == "stale"
    assert receiver.receive(8, 0, "new-answer") == "accepted"
    assert receiver.ready == ["new-answer"]
    small = ChunkReceiver(epoch=1, capacity=2)
    assert small.receive(1, 1, "second") == "accepted"
    assert small.receive(1, 2, "too-far-ahead") == "backpressure"
    assert small.receive(1, 0, "first") == "accepted"
    assert small.ready == ["first", "second"]
    return {"ordered_blocks_before_cancel": 4, "active_epoch": receiver.epoch,
            "late_output_dropped": True}


class BufferLifetime:
    """Logical cancellation cannot release a buffer with outstanding readers."""

    def __init__(self):
        self.readers = {"gpu", "transfer"}
        self.cancelled = False
        self.freed = False

    def release(self):
        if not self.cancelled or self.readers:
            raise RuntimeError("buffer still live")
        self.freed = True


def lifetime_and_backpressure():
    buffer = BufferLifetime()
    buffer.cancelled = True
    blocked = 0
    for completed in (None, "gpu", "transfer"):
        if completed:
            buffer.readers.remove(completed)
        try:
            buffer.release()
        except RuntimeError:
            blocked += 1
        else:
            assert not buffer.readers
    assert blocked == 2 and buffer.freed

    queued, reserved, capacity = 3, 1, 4
    assert queued < capacity  # This incomplete check would admit too much.
    assert queued + reserved == capacity
    admit_one_more = queued + reserved + 1 <= capacity
    assert not admit_one_more
    reserved -= 1
    queued += 1
    assert queued + reserved == 4
    queued -= 1  # One block consumed; now reserve a replacement.
    assert queued + reserved + 1 <= capacity
    reserved += 1
    assert (queued + reserved) * 9600 == 38400
    reserved -= 1  # Cancelled production releases reservation after drain.
    queued = 0
    assert queued + reserved == 0
    return {"premature_release_blocked": blocked, "queue_budget_bytes": 38400}


def training_objectives():
    text_losses, audio_losses = [.8, 1.2], [2.] * 8
    pooled = sum(text_losses + audio_losses) / 10
    balanced = .5 * sum(text_losses) / 2 + .5 * sum(audio_losses) / 8
    assert isclose(pooled, 1.8) and isclose(balanced, 1.5)
    # Adding padding changes neither sum nor the valid-position denominator.
    losses = text_losses + audio_losses + [999., 999.]
    mask = [1] * 10 + [0, 0]
    assert isclose(sum(x*m for x, m in zip(losses, mask)) / sum(mask), pooled)

    def task_mean(w):
        return .5 * ((w-1)**2 + (w+1)**2)

    def index_mean(w):
        return (2*(w-1)**2 + 8*(w+1)**2) / 10

    w, eps = .5, 1e-5
    assert isclose(task_mean(w), 1.25)
    assert isclose(index_mean(w), 1.85)
    for objective, derivative in ((task_mean, 1.), (index_mean, 2.2)):
        numerical = (objective(w+eps) - objective(w-eps)) / (2*eps)
        assert isclose(numerical, derivative, rel_tol=1e-8)

    def projector_loss(theta, weight):
        return (weight * theta * 3 - 1)**2 / 2

    assert projector_loss(2, .5) == 2
    dtheta = (projector_loss(2+eps, .5) - projector_loss(2-eps, .5)) / (2*eps)
    dw = (projector_loss(2, .5+eps) - projector_loss(2, .5-eps)) / (2*eps)
    assert isclose(dtheta, 3, rel_tol=1e-8)
    assert isclose(dw, 12, rel_tol=1e-8)
    return {"pooled_loss": pooled, "balanced_loss": balanced,
            "balanced_gradient": 1, "index_weighted_gradient": 2.2,
            "encoder_gradient": 3, "projector_gradient": 12}


def media_iteration():
    latent_bytes = 64 * 64 * 4 * 2
    positions = (64 // 2)**2
    assert latent_bytes == 32768 and positions == 1024
    assert positions * 1152 * 2 / 2**20 == 2.25
    larger_positions = (128 // 2)**2
    assert larger_positions / positions == 4
    assert larger_positions**2 / positions**2 == 16
    value, trajectory = -1., [-1.]
    for _ in range(4):
        value += .25 * 3
        trajectory.append(value)
    assert trajectory == [-1., -.25, .5, 1.25, 2.]
    checkpoint = {"latent": trajectory[2], "completed_steps": 2, "dt": .25}
    restored = checkpoint["latent"]
    for _ in range(checkpoint["completed_steps"], 4):
        restored += checkpoint["dt"] * 3
    assert restored == trajectory[-1]
    assert -1 + 2 * .25 * 3 != restored  # Seed alone cannot resume mid-run.
    assert (2.5 - 3)**2 == .25
    assert 50 + 20*40 + 30 + 20 == 900
    assert 50 + 10*40 + 30 + 20 == 500
    step7_end, step8_end, cancelled_at = 50 + 7*40, 50 + 8*40, 340
    assert step7_end < cancelled_at < step8_end
    return {"latent_bytes": latent_bytes, "patch_positions": positions,
            "flow_trajectory": trajectory, "restored_latent": restored,
            "cancel_ms": cancelled_at, "earliest_free_ms": step8_end}


def capacity_and_metrics():
    service_ms = [40, 100, 80, 20]
    rates = []
    for replicas in ([1, 1, 1, 1], [1, 2, 1, 1], [1, 2, 2, 1], [1, 3, 2, 1]):
        rates.append(min(n*1000/work for n, work in zip(replicas, service_ms)))
    assert rates == [10, 12.5, 20, 25]
    assert isclose(512000 / 10e9 * 1e6, 51.2)
    assert (1 + 1 + 0) / 10 == .2
    assert 60 / 100 == .6 and 60 / 10 == 6
    assert isclose(60 / 860, .06976744186046512)
    return {"capacity_requests_per_second": rates, "wer": .2,
            "qualified_fraction_of_arrivals": .6, "goodput_requests_per_second": 6}


def verify_source(root):
    manifest_path = Path(__file__).resolve().parents[1] / "reference/omni_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("Source file differs from fixed snapshot: " + str(path))
    return {"expected_commit": manifest["commit"], "matched_files": len(manifest["files"]),
            "scope": "file identity only; no upstream execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    checks = (representations, playback_timeline, chunk_identity, lifetime_and_backpressure,
              training_objectives, media_iteration, capacity_and_metrics)
    report = {"evidence_kind": "CPU teaching arithmetic and state simulation",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": {check.__name__: check() for check in checks}}
    if args.source_root:
        report["source_identity"] = verify_source(args.source_root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for name in report["checks"]:
            print("PASS " + name)
        if args.source_root:
            print("PASS fixed source file identities")
        print("All Omni teaching checks passed; no model/GPU/audio-device execution.")


if __name__ == "__main__":
    main()
