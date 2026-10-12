"""Deterministic teaching models for training, queues and capacity."""

from collections import deque
from math import isclose


def training_and_storage():
    theta = 0.0
    for labels in ([0, 2], [0, 2]):
        theta -= 0.1 * sum(theta-y for y in labels) / len(labels)
    assert isclose(theta, 0.19)
    assert isclose(-0.1 * sum(-y for y in [0, 2, 0, 2]) / 4, 0.1)
    assert isclose(1.4 / 1.2, 7/6)
    assert isclose(1e6 / (0.005 + 1e6/1e9), 1e9/6)
    assert isclose(80 / (2 - 1.2), 100)
    assert sum([12, 20, 4, 10]) == 46 and max([12, 20, 4, 10]) == 20


def queue_and_capacity():
    fcfs, now = {}, 0
    for name, work in (("A", 8), ("B", 1), ("C", 1)):
        now += work
        fcfs[name] = now
    pending = deque([("A", 8), ("B", 1), ("C", 1)])
    rr, now = {}, 0
    while pending:
        name, remaining = pending.popleft()
        now += 1
        if remaining == 1:
            rr[name] = now
        else:
            pending.append((name, remaining-1))
    assert fcfs == {"A": 8, "B": 9, "C": 10}
    assert rr == {"B": 2, "C": 3, "A": 10}
    assert sum(fcfs.values()) / 3 == 9 and sum(rr.values()) / 3 == 5
    assert sum(t <= 4 for t in rr.values()) / now == 0.2
    finish, latencies = 0, []
    for arrival in range(5):
        finish = max(finish, arrival) + 2
        latencies.append(finish-arrival)
    assert latencies == [2, 3, 4, 5, 6]
    mean_square = (0.2**2 + 1.8**2)/2
    assert isclose(0.5 * mean_square / (2*(1-0.5)), 0.82)
    assert (14-10)*30 == 120 and 120/(20-14) == 20
    print({"FCFS": fcfs, "round_robin": rr, "open_loop_latency": latencies})


if __name__ == "__main__":
    training_and_storage()
    queue_and_capacity()
    print("PASS training, storage, queue and capacity teaching models")
