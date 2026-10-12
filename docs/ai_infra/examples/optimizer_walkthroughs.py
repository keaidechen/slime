"""CPU arithmetic references for AdamW and Muon teaching articles.

Run: python3 docs/ai_infra/examples/optimizer_walkthroughs.py
Uses analytic answers and a spectral oracle; does not exercise GPU optimizers.
"""

from math import sqrt

from parallel_runtime_walkthroughs import add, close, mm, transpose


def adam_history_and_decay():
    gradients = [[2.0, -1.0], [0.0, 3.0]]
    sgd, momentum, adam, adamw = ([1.0, -2.0] for _ in range(4))
    velocity, m, v = ([0.0, 0.0] for _ in range(3))
    for step, gradient in enumerate(gradients, 1):
        for i, g in enumerate(gradient):
            sgd[i] -= 0.1 * g
            velocity[i] = 0.5 * velocity[i] + g
            momentum[i] -= 0.1 * velocity[i]
            m[i] = 0.5 * m[i] + 0.5 * g
            v[i] = 0.5 * v[i] + 0.5 * g * g
            direction = (m[i] / (1 - 0.5**step)) / sqrt(v[i] / (1 - 0.5**step))
            adam[i] -= 0.1 * direction
            adamw[i] = 0.99 * adamw[i] - 0.1 * direction
        if step == 1:
            close(adam, [0.9, -1.9])
            close(adamw, [0.89, -1.88])
    close(m, [0.5, 1.25])
    close(v, [1.0, 4.75])
    close(sgd, [0.8, -2.2])
    close(momentum, [0.7, -2.15])
    close(adam, [0.9 - 0.1 / sqrt(3), -1.9 - 0.5 / sqrt(57)])
    close(adamw, [0.8811 - 0.1 / sqrt(3), -1.8612 - 0.5 / sqrt(57)])

    # Zero task gradient distinguishes coupled L2 from decoupled decay.
    coupled = [w - 0.1 * (0.1 * w) / (0.1 * w + 1e-8) for w in (1.0, 2.0)]
    decoupled = [0.99 * w for w in (1.0, 2.0)]
    close(coupled, [0.9, 1.9])
    close(decoupled, [0.99, 1.98])
    assert abs(coupled[0] - decoupled[0]) > 0.08


def cubic_polar(matrix, steps):
    """Teaching iteration, not the finite quintic used by production Muon."""
    norm = sqrt(sum(x * x for row in matrix for x in row))
    if norm == 0:
        return [row[:] for row in matrix]
    x = [[value / norm for value in row] for row in matrix]
    for _ in range(steps):
        cubic = mm(mm(x, transpose(x)), x)
        x = [[1.5 * a - 0.5 * b for a, b in zip(row, cube)]
             for row, cube in zip(x, cubic)]
    return x


def matrix_direction_and_spectral_oracle():
    gradient = [[2.0, 1.0], [1.0, 2.0]]
    high, low = 3 / sqrt(10), 1 / sqrt(10)
    # Analytic eigenvectors are fixed; scalar evolution independently checks
    # the matrix multiplication path, including every intermediate iterate.
    for step in range(9):
        expected = [[(high + low) / 2, (high - low) / 2],
                    [(high - low) / 2, (high + low) / 2]]
        close(cubic_polar(gradient, step), expected)
        high, low = (1.5 * x - 0.5 * x**3 for x in (high, low))
    close(cubic_polar(gradient, 8), [[1, 0], [0, 1]])
    close(cubic_polar(gradient, 5),
          [[0.9987222260915564, 0.0012777739084436],
           [0.0012777739084436, 0.9987222260915564]])
    # Rectangular, rank-deficient, and zero cases forbid a blanket Q^T Q = I.
    close(cubic_polar([[3, 4]], 8), [[0.6, 0.8]])
    close(cubic_polar([[3, 0], [0, 0]], 8), [[1, 0], [0, 0]])
    close(cubic_polar([[0, 0], [0, 0]], 8), [[0, 0], [0, 0]])
    close(3.4445 - 4.7750 + 2.0315, 0.701)


def sharding_counterexample():
    gradient = [[2.0, 1.0], [1.0, 2.0]]
    local = [cubic_polar([row], 8)[0] for row in gradient]
    close(local, [[2 / sqrt(5), 1 / sqrt(5)], [1 / sqrt(5), 2 / sqrt(5)]])
    close(mm(local, transpose(local)), [[1, 0.8], [0.8, 1]])
    full = cubic_polar(gradient, 8)
    assert abs(local[0][1] - full[0][1]) > 0.4

    # Column shards with a shared normalization and summed Gram reconstruct
    # the full update; local-only normalization is deliberately not used.
    shards = [[[row[j] / sqrt(10)] for row in gradient] for j in range(2)]
    for _ in range(5):
        gram = add(*(mm(shard, transpose(shard)) for shard in shards))
        updated = []
        for shard in shards:
            product = mm(gram, shard)
            updated.append([[1.5 * row[0] - 0.5 * cube[0]]
                            for row, cube in zip(shard, product)])
        shards = updated
    assembled = [[shards[j][i][0] for j in range(2)] for i in range(2)]
    close(assembled, cubic_polar(gradient, 5))


if __name__ == "__main__":
    for check in (adam_history_and_decay, matrix_direction_and_spectral_oracle,
                  sharding_counterexample):
        check()
        print(f"PASS {check.__name__}")
    print("All optimizer references passed (CPU, analytic teaching models).")
