"""Single-head attention math lesson; Python standard library, CPU, float64.

This demonstrates the recurrence, not CUDA layouts, FP16 rounding, or performance.
Run from the repository root:
    python3 docs/ai_infra/13_Source_Analysis/flash_attention/examples/online_attention.py
"""

import math
import random


def visible(i, j, nq, nk, causal):
    """FA2's bottom-right causal alignment, also for rectangular attention."""
    return not causal or j <= i + nk - nq


def dense_attention(q, k, v, causal=False):
    """Independent reference: construct and normalize each full score row."""
    scale = len(q[0]) ** -0.5
    out, lse = [], []
    for i, qi in enumerate(q):
        scores = [
            sum(x * y for x, y in zip(qi, kj)) * scale
            if visible(i, j, len(q), len(k), causal) else -math.inf
            for j, kj in enumerate(k)
        ]
        maximum = max(scores)
        if maximum == -math.inf:
            out.append([0.0] * len(v[0]))
            lse.append(-math.inf)
            continue
        weights = [math.exp(s - maximum) for s in scores]
        denominator = sum(weights)
        probabilities = [w / denominator for w in weights]
        out.append([
            sum(p * vj[t] for p, vj in zip(probabilities, v))
            for t in range(len(v[0]))
        ])
        lse.append(maximum + math.log(denominator))
    return out, lse


def online_attention(q, k, v, tile_size=2, causal=False, reverse=False, trace=False):
    """Keep only a tile of scores and m/l/a for each query row.

    Inputs are nonempty, rectangular Python lists with compatible dimensions.
    We use -inf as the empty-row LSE convention in this teaching implementation.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    scale = len(q[0]) ** -0.5
    starts = list(range(0, len(k), tile_size))
    if reverse:
        starts.reverse()
    out, lse = [], []
    for i, qi in enumerate(q):
        m, l = -math.inf, 0.0
        a = [0.0] * len(v[0])
        for start in starts:
            indices = [
                j for j in range(start, min(start + tile_size, len(k)))
                if visible(i, j, len(q), len(k), causal)
            ]
            if not indices:
                continue
            scores = [sum(x * y for x, y in zip(qi, k[j])) * scale for j in indices]
            new_m = max(m, max(scores))
            correction = 0.0 if m == -math.inf else math.exp(m - new_m)
            weights = [math.exp(s - new_m) for s in scores]
            l = correction * l + sum(weights)
            a = [
                correction * a[t] + sum(w * v[j][t] for w, j in zip(weights, indices))
                for t in range(len(a))
            ]
            m = new_m
            if trace:
                print(f"  row={i}, keys={indices}, m={m:.6f}, l={l:.6f}, a={a}")
        out.append([x / l for x in a] if l else [0.0] * len(a))
        lse.append(m + math.log(l) if l else -math.inf)
    return out, lse


def combine_partials(partials):
    """Merge (normalized output, LSE) pairs from disjoint K/V splits."""
    out, lse = [], []
    for i in range(len(partials[0][0])):
        maximum = max(local_lse[i] for _, local_lse in partials)
        if maximum == -math.inf:
            out.append([0.0] * len(partials[0][0][i]))
            lse.append(-math.inf)
            continue
        weights = [math.exp(local_lse[i] - maximum) for _, local_lse in partials]
        denominator = sum(weights)
        out.append([
            sum(w * local_out[i][t] for w, (local_out, _) in zip(weights, partials)) / denominator
            for t in range(len(partials[0][0][i]))
        ])
        lse.append(maximum + math.log(denominator))
    return out, lse


def reference_backward(q, k, v, dout, causal=False):
    """Teaching gradient formulas; materializes row probabilities on CPU."""
    out, lse = dense_attention(q, k, v, causal)
    scale = len(q[0]) ** -0.5
    dq = [[0.0] * len(q[0]) for _ in q]
    dk = [[0.0] * len(k[0]) for _ in k]
    dv = [[0.0] * len(v[0]) for _ in v]
    for i, qi in enumerate(q):
        if lse[i] == -math.inf:
            continue
        row_d = sum(o * do for o, do in zip(out[i], dout[i]))
        for j, kj in enumerate(k):
            if not visible(i, j, len(q), len(k), causal):
                continue
            score = sum(x * y for x, y in zip(qi, kj)) * scale
            p = math.exp(score - lse[i])
            dp = sum(do * value for do, value in zip(dout[i], v[j]))
            ds = p * (dp - row_d)
            for t in range(len(qi)):
                dq[i][t] += scale * ds * kj[t]
                dk[j][t] += scale * ds * qi[t]
            for t in range(len(v[j])):
                dv[j][t] += p * dout[i][t]
    return dq, dk, dv


def check_close(actual, expected, tolerance=1e-12):
    assert len(actual) == len(expected)
    for x, y in zip(actual, expected):
        if isinstance(x, list):
            check_close(x, y, tolerance)
        elif x != y:  # Equal infinities are accepted for empty rows.
            assert math.isfinite(x) and math.isfinite(y), (x, y)
            assert abs(x - y) <= tolerance, (x, y)


def self_check():
    rng = random.Random(7)

    def matrix(rows, columns):
        return [[rng.uniform(-1, 1) for _ in range(columns)] for _ in range(rows)]

    cases = 0
    for nq, nk in [(1, 1), (2, 5), (5, 2), (7, 9)]:
        q, k, v = matrix(nq, 3), matrix(nk, 3), matrix(nk, 2)
        for causal in (False, True):
            expected = dense_attention(q, k, v, causal)
            for tile in (1, 2, 4, 16):
                for reverse in (False, True):
                    actual = online_attention(q, k, v, tile, causal, reverse)
                    check_close(actual[0], expected[0])
                    check_close(actual[1], expected[1])
                    cases += 1
        if nk > 1:
            mid = nk // 2
            partials = [dense_attention(q, k[:mid], v[:mid]), dense_attention(q, k[mid:], v[mid:])]
            actual = combine_partials(partials)
            expected = dense_attention(q, k, v)
            check_close(actual[0], expected[0])
            check_close(actual[1], expected[1])
    # Scores around one million: subtracting the running maximum avoids overflow.
    q, k, v = [[1000.0]], [[1000.0], [1001.0], [-1000.0]], [[1.0], [2.0], [3.0]]
    for reverse in (False, True):
        actual = online_attention(q, k, v, 1, reverse=reverse)
        expected = dense_attention(q, k, v)
        check_close(actual[0], expected[0])
        check_close(actual[1], expected[1])

    # Central finite differences independently check the derivative formulas.
    q, k, v, dout = matrix(3, 2), matrix(2, 2), matrix(2, 2), matrix(3, 2)
    max_gradient_error = 0.0
    for causal in (False, True):
        analytical = reference_backward(q, k, v, dout, causal)

        def loss():
            out, _ = dense_attention(q, k, v, causal)
            return sum(x * y for row, grad in zip(out, dout) for x, y in zip(row, grad))

        for tensor, gradient in zip((q, k, v), analytical):
            for i, row in enumerate(tensor):
                for t, value in enumerate(row):
                    epsilon = 1e-6
                    tensor[i][t] = value + epsilon
                    plus = loss()
                    tensor[i][t] = value - epsilon
                    minus = loss()
                    tensor[i][t] = value
                    numerical = (plus - minus) / (2 * epsilon)
                    error = abs(numerical - gradient[i][t])
                    max_gradient_error = max(max_gradient_error, error)
                    assert error < 1e-8, (i, t, numerical, gradient[i][t])
    print(f"PASS: {cases} tiled forward cases, split merge, and large-score cases")
    print(f"PASS: backward finite differences, max absolute error = {max_gradient_error:.3e}")


def main():
    q = [[1.0]]
    k = [[math.log(x)] for x in (1, 2, 4, 8)]
    v = [[1.0], [2.0], [3.0], [4.0]]
    print("Worked example: scores=[0, ln(2), ln(4), ln(8)], V=[1,2,3,4]")
    out, _ = online_attention(q, k, v, trace=True)
    print(f"Output = {out[0][0]:.9f}; expected 49/15 = {49 / 15:.9f}")
    check_close(out, [[49 / 15]])
    self_check()


if __name__ == "__main__":
    main()
