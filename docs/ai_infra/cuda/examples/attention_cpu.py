"""Progressive attention lab using only Python's standard library.

Run normally (not python -O): assertions are the exercise's correctness checks.
This does not simulate CUDA concurrency or floating-point instruction rounding.
"""

import math
import random


def softmax(row):
    maximum = max(row)
    if maximum == -math.inf:
        return [0.0] * len(row)  # Our explicit convention for a fully masked row.
    weights = [math.exp(x - maximum) for x in row]
    total = sum(weights)
    return [x / total for x in weights]


def materialized_attention(q, k, v, limits):
    scale = 1 / math.sqrt(len(q[0]))
    scores = [
        [sum(x * y for x, y in zip(qi, kj)) * scale if j < limit else -math.inf for j, kj in enumerate(k)]
        for qi, limit in zip(q, limits)
    ]
    probabilities = [softmax(row) for row in scores]
    return [[sum(p[j] * v[j][c] for j in range(len(k))) for c in range(len(v[0]))] for p in probabilities]


def online_row(scores, values, tile, reverse=False, trace=False, rescale=True):
    maximum, denominator = -math.inf, 0.0
    numerator = [0.0] * len(values[0])
    starts = list(range(0, len(scores), tile))
    if reverse:
        starts.reverse()
    for start in starts:
        block = scores[start : start + tile]
        block_max = max(block)
        if block_max == -math.inf:
            continue
        new_max = max(maximum, block_max)
        alpha = math.exp(maximum - new_max) if rescale else 1.0
        weights = [math.exp(s - new_max) for s in block]
        denominator = alpha * denominator + sum(weights)
        numerator = [
            alpha * old + sum(p * values[start + j][c] for j, p in enumerate(weights))
            for c, old in enumerate(numerator)
        ]
        maximum = new_max
        if trace:
            print(f"tile@{start}: m={maximum:.6f}, alpha={alpha:.6f}, l={denominator:.6f}, a={numerator}")
    if denominator == 0:
        return [0.0] * len(numerator)
    return [x / denominator for x in numerator]


def online_attention(q, k, v, limits, tile, reverse=False):
    result = []
    scale = 1 / math.sqrt(len(q[0]))
    for qi, limit in zip(q, limits):
        # Materialized here to keep the CPU proof readable; CUDA computes one tile at a time.
        scores = [sum(x * y for x, y in zip(qi, kj)) * scale if j < limit else -math.inf for j, kj in enumerate(k)]
        result.append(online_row(scores, v, tile, reverse))
    return result


def check_close(actual, expected):
    assert len(actual) == len(expected)
    for row, ref in zip(actual, expected):
        assert len(row) == len(ref)
        for x, y in zip(row, ref):
            assert math.isfinite(x) and math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10), (x, y)


def check_layouts():
    # One m16n8 MMA accumulator: each lane owns 2 rows x 2 columns.
    coordinates = []
    for lane in range(32):
        for i in range(2):
            for j in range(2):
                coordinates.append((lane // 4 + i * 8, (lane % 4) * 2 + j))
    assert len(coordinates) == len(set(coordinates)) == 16 * 8
    assert set(coordinates) == {(r, c) for r in range(16) for c in range(8)}
    # Fixed FA2 configuration in chapter 09, not an arbitrary MMA layout.
    full = []
    for tid in range(128):
        warp, lane = divmod(tid, 32)
        for mi in range(2):
            for i in range(2):
                for nj in range(16):
                    for j in range(2):
                        full.append((warp * 16 + lane // 4 + i * 8 + mi * 64, (lane % 4) * 2 + j + nj * 8))
    assert len(full) == len(set(full)) == 128 * 128
    assert set(full) == {(r, c) for r in range(128) for c in range(128)}
    print("PASS layouts: 32 lanes cover 16x8; fixed FA2 128 threads cover 128x128")


def main():
    scores = [math.log(x) for x in (1, 2, 4, 8)]
    values = [[1.0], [2.0], [3.0], [4.0]]
    correct = online_row(scores, values, tile=2, trace=True)
    check_close([correct], [[49 / 15]])
    wrong = online_row(scores, values, tile=2, rescale=False)
    assert abs(wrong[0] - correct[0]) > 0.1
    print(f"Missing-rescale counterexample: wrong={wrong[0]:.6f}, correct={correct[0]:.6f}")

    base = softmax([0.0, 1.0, -2.0])
    check_close([base], [softmax([1000.0, 1001.0, 998.0])])
    assert softmax([-math.inf] * 3) == [0.0] * 3
    rng = random.Random(2026)
    comparisons = 0
    for nq, nk, d, dv in ((1, 1, 1, 1), (3, 7, 5, 3), (4, 33, 8, 9), (5, 129, 7, 17)):
        q = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(nq)]
        k = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(nk)]
        v = [[rng.uniform(-2, 2) for _ in range(dv)] for _ in range(nk)]
        # Dense, entirely empty, and rows with different valid prefix lengths.
        for limits in ([nk] * nq, [0] * nq, [min(nk, i * 17) for i in range(nq)]):
            reference = materialized_attention(q, k, v, limits)
            for tile in (1, 2, 7, 32, 128):
                for reverse in (False, True):
                    check_close(online_attention(q, k, v, limits, tile, reverse), reference)
                    comparisons += 1
    # Equal-length causal attention: row i may read keys 0..i.
    q = [[1.0], [2.0], [3.0]]
    k, v = [[0.0], [1.0], [2.0]], [[10.0], [20.0], [30.0]]
    check_close(online_attention(q, k, v, [1, 2, 3], 2), materialized_attention(q, k, v, [1, 2, 3]))
    check_layouts()
    print(f"PASS softmax/attention: {comparisons} partition/order comparisons plus shift, empty and causal cases")
    print("CPU proofs only; CUDA compilation, races and device accuracy require GPU checks.")


if __name__ == "__main__":
    main()
