"""Bounded CUDA/Triton correctness and timing lesson for contiguous float32 vectors."""
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, z, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < n
    xv = tl.load(x + offsets, mask=active, other=0)
    yv = tl.load(y + offsets, mask=active, other=0)
    tl.store(z + offsets, xv + yv, mask=active)


def elapsed_ms(fn, repeats=15, inner=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / inner)
    return statistics.median(samples), min(samples), max(samples)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('This lesson requires CUDA and Triton')
    print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__)
    for n in (1, 257, 1000, 2**20):
        for seed in (0, 1):
            torch.manual_seed(seed)
            x = torch.randn(n, device='cuda', dtype=torch.float32)
            y = torch.randn_like(x)
            z = torch.empty_like(x)
            add_kernel[(triton.cdiv(n, 256),)](x, y, z, n, BLOCK=256)
            torch.testing.assert_close(z, x + y)
        baseline = torch.empty_like(x)
        eager = lambda: torch.add(x, y, out=baseline)
        custom = lambda: add_kernel[(triton.cdiv(n, 256),)](x, y, z, n, BLOCK=256)
        print({'n': n, 'torch_ms_median_min_max': elapsed_ms(eager),
               'triton_ms_median_min_max': elapsed_ms(custom)})


if __name__ == '__main__':
    main()
