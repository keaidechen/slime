"""CPU-only index/algorithm walkthrough; does not emulate GPU races or scheduling."""


def block_partials(values, block_size):
    assert block_size > 0 and block_size & (block_size - 1) == 0
    partials = []
    for base in range(0, len(values), block_size):
        shared = values[base : base + block_size]
        shared += [0.0] * (block_size - len(shared))
        step = block_size // 2
        while step:
            # Each phase represents work between two block barriers.
            for t in range(step):
                shared[t] += shared[t + step]
            step //= 2
        partials.append(shared[0])
    return partials


def tiled_matmul(a, b, m, n, k, tile):
    c = [0.0] * (m * n)
    for by in range((m + tile - 1) // tile):
        for bx in range((n + tile - 1) // tile):
            acc = [[0.0] * tile for _ in range(tile)]
            for base in range(0, k, tile):
                sa = [[0.0] * tile for _ in range(tile)]
                sb = [[0.0] * tile for _ in range(tile)]
                for ty in range(tile):
                    for tx in range(tile):
                        row, col = by * tile + ty, bx * tile + tx
                        if row < m and base + tx < k:
                            sa[ty][tx] = a[row * k + base + tx]
                        if base + ty < k and col < n:
                            sb[ty][tx] = b[(base + ty) * n + col]
                for ty in range(tile):
                    for tx in range(tile):
                        for j in range(tile):
                            acc[ty][tx] += sa[ty][j] * sb[j][tx]
            for ty in range(tile):
                for tx in range(tile):
                    row, col = by * tile + ty, bx * tile + tx
                    if row < m and col < n:
                        c[row * n + col] = acc[ty][tx]
    return c


def main():
    # Chapter 1's fictional SM. This is a resource budget, not a GPU simulator.
    # Ignore allocation granularity, per-block limits, SM partitions and reservations.
    for threads, registers, shared_kib, expected_blocks, expected_percent in (
        (256, 32, 16, 4, 50.0),
        (256, 32, 32, 2, 25.0),
        (256, 96, 16, 2, 25.0),
        (128, 32, 16, 4, 25.0),
    ):
        warps = (threads + 31) // 32
        blocks = min(2048 // threads, 64 // warps, 65536 // (threads * registers), 64 // shared_kib, 16)
        percent = 100 * blocks * warps / 64
        assert (blocks, percent) == (expected_blocks, expected_percent)
        print(f"Toy SM: T={threads}, R={registers}, S={shared_kib} KiB -> {blocks} blocks, {percent:.0f}% occupancy")
    coordinates = [(x, y) for y in range(16) for x in range(16)]
    assert [(x + 16 * y) // 32 for x, y in coordinates[:32]] == [0] * 32
    assert divmod(5 + 16 * 2, 32) == (1, 5)
    print("PASS hardware mapping: fictional resource budget and 16x16 thread numbering")

    for n in (1, 31, 32, 255, 256, 257, 1003):
        indices = [
            block * 256 + thread
            for block in range((n + 255) // 256)
            for thread in range(256)
            if block * 256 + thread < n
        ]
        assert indices == list(range(n))
        values = [(i % 13 - 6) * 0.125 for i in range(n)]
        partials = block_partials(values, 256)
        assert partials == [sum(values[i : i + 256]) for i in range(0, n, 256)]
        assert sum(partials) == sum(values)
    for m, n, k in ((1, 1, 1), (16, 16, 16), (17, 19, 23), (31, 7, 33)):
        a = [(i % 11 - 5) * 0.125 for i in range(m * k)]
        b = [(i % 7 - 3) * 0.25 for i in range(k * n)]
        expected = [sum(a[r * k + j] * b[j * n + c] for j in range(k)) for r in range(m) for c in range(n)]
        assert tiled_matmul(a, b, m, n, k, 16) == expected
    print("PASS CPU walkthrough: indices, block partials, tiled matmul")
    print("This checks algorithms only, not CUDA compilation, synchronization or performance.")


if __name__ == "__main__":
    main()
