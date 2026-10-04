"""Pure-Python lesson: local means, global mean, and DDP gradient scaling."""
from math import isclose


def main():
    losses = [[1.0, 3.0], [9.0]]
    mean_of_means = sum(sum(x) / len(x) for x in losses) / len(losses)
    global_mean = sum(map(sum, losses)) / sum(map(len, losses))
    assert isclose(mean_of_means, 5.5)
    assert isclose(global_mean, 13 / 3)
    # Choose per-token l_i(w) = (w-a_i)^2 / 2. Its derivative is w-a_i.
    labels, w = [[0.0, 2.0], [8.0]], 1.0
    count, ranks = sum(map(len, labels)), len(labels)
    reference_grad = sum(w-a for xs in labels for a in xs) / count
    local_scaled_grads = [ranks * sum(w-a for a in xs) / count for xs in labels]
    ddp_averaged_grad = sum(local_scaled_grads) / ranks
    assert isclose(reference_grad, ddp_averaged_grad)
    print({'mean_of_means': mean_of_means, 'global_mean': global_mean,
           'reference_gradient': reference_grad,
           'DDP_average_of_scaled_gradients': ddp_averaged_grad})


if __name__ == '__main__':
    main()
