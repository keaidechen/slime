"""Local CPU lesson: bounded submissions with ray.wait; no existing cluster."""
import argparse
import time

import ray


@ray.remote(num_cpus=1)
def produce(i):
    time.sleep(0.15 if i % 4 == 0 else 0.03)
    return {'sample_id': i, 'tokens': 10 + i}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-pending', type=int, default=4)
    args = parser.parse_args()
    if args.max_pending < 1:
        parser.error('--max-pending must be positive')
    ray.init(address='local', num_cpus=2)
    pending, results = [], []
    start = time.perf_counter()
    try:
        for i in range(12):
            if len(pending) >= args.max_pending:
                ready, pending = ray.wait(pending, num_returns=1)
                results.extend(ray.get(ready))
            pending.append(produce.remote(i))
            assert len(pending) <= args.max_pending
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            results.extend(ray.get(ready))
        assert sorted(x['sample_id'] for x in results) == list(range(12))
        print('completion order:', [x['sample_id'] for x in results])
        print('elapsed seconds:', time.perf_counter() - start)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
