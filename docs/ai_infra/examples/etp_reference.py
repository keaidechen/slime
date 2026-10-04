"""Manual CPU teaching lab: ETP math and EP routing, without distributed backends.

Run from the repository root: python3 docs/ai_infra/examples/etp_reference.py
Uses independent unsharded forward/finite differences as the numerical reference.
It does not execute Megatron, NCCL, Transformer Engine, or optimizer sharding.
"""

from itertools import product

from parallel_runtime_walkthroughs import close, finite_difference, mm


INPUT = [[1.0, 2.0], [3.0, 4.0]]
W1 = [[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 0.0]]
W2 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]


def reference(x, a, b):
    """Whole expert, no shard or routing logic."""
    return mm([[max(0.0, z) for z in row] for row in mm(x, a)], b)


def sharded_expert(x, a, b, degree, output_grad):
    """Column FC1 / row FC2 and backward; SUM is simulated with Python addition."""
    n, h, f = len(x), len(a), len(b)
    assert f % degree == 0
    y = [[0.0] * h for _ in range(n)]
    dx = [[0.0] * h for _ in range(n)]
    da = [[0.0] * f for _ in range(h)]
    db = [[0.0] * h for _ in range(f)]
    for shard in range(degree):
        for j in range(shard * f // degree, (shard + 1) * f // degree):
            for t in range(n):
                preact = sum(x[t][i] * a[i][j] for i in range(h))
                z = max(0.0, preact)
                dz = sum(output_grad[t][o] * b[j][o] for o in range(h))
                dz *= float(preact > 0.0)
                for o in range(h):
                    y[t][o] += z * b[j][o]
                    db[j][o] += z * output_grad[t][o]
                for i in range(h):
                    dx[t][i] += dz * a[i][j]
                    da[i][j] += x[t][i] * dz
    return y, dx, da, db


def etp_forward_backward():
    for degree in (1, 2, 4):
        y, dx, da, db = sharded_expert(INPUT, W1, W2, degree, [[1.0, 1.0]] * 2)
        close(y, reference(INPUT, W1, W2))
        close(y, [[8, 3], [22, 5]])
        close(dx, [[5, 3], [5, 3]])
        close(da, [[4, 4, 8, 4], [6, 6, 12, 6]])
        close(db, [[4, 4], [6, 6], [10, 10], [8, 8]])


def etp_finite_difference():
    # Nonuniform upstream gradient catches accidental averaging and token mixups.
    g = [[0.3, -0.7], [1.2, 0.4]]
    _, dx, da, db = sharded_expert(INPUT, W1, W2, 2, g)

    def loss(x, a, b):
        return sum(v * w for row, weights in zip(reference(x, a, b), g)
                   for v, w in zip(row, weights))

    def rows(values, width):
        return [values[i:i + width] for i in range(0, len(values), width)]

    close(sum(dx, []), finite_difference(lambda v: loss(rows(v, 2), W1, W2), sum(INPUT, [])))
    close(sum(da, []), finite_difference(lambda v: loss(INPUT, rows(v, 4), W2), sum(W1, [])))
    close(sum(db, []), finite_difference(lambda v: loss(INPUT, W1, rows(v, 2)), sum(W2, [])))


def ep_etp_dispatch_combine():
    x = INPUT + [[2.0, 1.0], [1.0, 3.0]]
    source_rank = [0, 0, 1, 1]
    # Four experts, top-2, EP=2; e3 empty, e0 hot, asymmetric send/recv splits.
    routes = [[(0, .7), (2, .3)], [(1, .6), (0, .4)],
              [(0, .9), (1, .1)], [(2, .8), (0, .2)]]
    matrices = [[[value * (e + 1) for value in row] for row in W1] for e in range(4)]
    queues = [[] for _ in range(4)]
    sends = [[0, 0], [0, 0]]
    expected = [[0.0, 0.0] for _ in x]
    for t, assignments in enumerate(routes):
        for e, probability in assignments:
            queues[e].append((t, probability))
            sends[source_rank[t]][e // 2] += 1
            for o, value in enumerate(reference([x[t]], matrices[e], W2)[0]):
                expected[t][o] += probability * value
    assert sends == [[3, 1], [3, 1]]
    receives = [[sends[source][dest] for source in range(2)] for dest in range(2)]
    assert receives == [[3, 3], [1, 1]]
    assert [len(q) for q in queues] == [4, 2, 2, 0]
    for degree in (1, 2, 4):
        actual = [[0.0, 0.0] for _ in x]
        actual_dx = [[0.0, 0.0] for _ in x]
        for e, queue in enumerate(queues):
            # Deliberately reverse arrival order. Original token IDs must survive.
            queue = list(reversed(queue))
            tokens = [x[t] for t, _ in queue]
            grads = [[p, p] for _, p in queue]  # loss=sum(weighted outputs)
            y, dx, _, _ = sharded_expert(tokens, matrices[e], W2, degree, grads)
            for i, (t, probability) in enumerate(queue):
                for o in range(2):
                    actual[t][o] += probability * y[i][o]
                    actual_dx[t][o] += dx[i][o]
        close(actual, expected)

        def moe_loss(flat_x):
            tokens = [flat_x[i:i + 2] for i in range(0, len(flat_x), 2)]
            return sum(p * sum(reference([tokens[t]], matrices[e], W2)[0])
                       for t, assignments in enumerate(routes) for e, p in assignments)

        close(sum(actual_dx, []), finite_difference(moe_loss, sum(x, [])))


def groups(sizes, varying):
    """Mixed-radix teaching model: first axis varies fastest, like RankGenerator."""
    result = {}
    for coord in product(*(range(size) for size in sizes)):
        rank, stride = 0, 1
        for value, size in zip(coord, sizes):
            rank += value * stride
            stride *= size
        fixed = tuple(value for axis, value in enumerate(coord) if axis not in varying)
        result.setdefault(fixed, []).append(rank)
    return sorted(sorted(members) for members in result.values())


def dense_expert_mesh():
    # Axes: tp, cp, ep, dp, pp. Dense and expert grids reuse the same 8 ranks.
    dense = [2, 2, 1, 2, 1]
    expert = [2, 1, 2, 2, 1]
    close(groups(dense, {0}), [[0, 1], [2, 3], [4, 5], [6, 7]])
    close(groups(expert, {0}), [[0, 1], [2, 3], [4, 5], [6, 7]])
    close(groups(expert, {2}), [[0, 2], [1, 3], [4, 6], [5, 7]])
    close(groups(expert, {3}), [[0, 4], [1, 5], [2, 6], [3, 7]])
    close(groups(expert, {4}), groups(dense, {4}))
    # ETP=1, EP=4, EDP=2 still consumes 8 ranks, with a different expert layout.
    folded = [1, 1, 4, 2, 1]
    close(groups(folded, {2}), [[0, 1, 2, 3], [4, 5, 6, 7]])
    close(groups(folded, {3}), [[0, 4], [1, 5], [2, 6], [3, 7]])


if __name__ == "__main__":
    for check in (etp_forward_backward, etp_finite_difference,
                  ep_etp_dispatch_combine, dense_expert_mesh):
        check()
        print(f"PASS {check.__name__}")
    print("CPU mathematical references passed; no distributed backend was executed.")
