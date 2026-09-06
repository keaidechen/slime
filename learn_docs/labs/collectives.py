"""Two-rank correctness lesson. Run with torchrun; not a bandwidth benchmark."""
import argparse
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('gloo', 'nccl'), default='gloo')
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    if args.backend == 'nccl':
        if not torch.cuda.is_available():
            raise RuntimeError('NCCL lesson requires a CUDA-enabled PyTorch and visible GPUs')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    dist.init_process_group(args.backend, timeout=timedelta(seconds=60))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size != 2:
            raise ValueError('This lesson requires exactly two workers')
        original = torch.full((2,), float(rank + 1), device=device)
        reduced = original.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(reduced, torch.full_like(reduced, 3.0))
        gathered = [torch.empty_like(original) for _ in range(size)]
        dist.all_gather(gathered, original)
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0]], device=device)
        torch.testing.assert_close(torch.stack(gathered), expected)
        print({'pid': os.getpid(), 'rank': rank, 'local_rank': local_rank,
               'world_size': size, 'device': str(device),
               'sum': reduced.cpu().tolist(),
               'gather': torch.stack(gathered).cpu().tolist()}, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
