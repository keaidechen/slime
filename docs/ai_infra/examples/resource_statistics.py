"""Resource arithmetic and exact small-sample statistics; no cluster required."""

from itertools import product
from math import ceil, comb, isclose, sqrt
from statistics import mean, stdev


def cluster_and_multimodal():
    free = [4, 4]
    assert sum(free) == 8 and max(free) < 8
    assert sum(slots//4 for slots in free) == 2
    assert 8*8*8 == 4*8*16 == 512
    assert 4*8*8 == 256
    assert 10*60/2 + 40 == 340
    visual_small, visual_large = (224//14)**2, (448//14)**2
    assert (visual_small, visual_large) == (256, 1024)
    assert (visual_large/visual_small)**2 == 16
    assert ((128+visual_large)/(128+visual_small))**2 == 9
    bytes_per_token = 2*32*8*128*2
    assert bytes_per_token == 128*1024
    assert [(128+n)*bytes_per_token//2**20 for n in (256, 1024, 16*256)] == [48, 144, 528]
    assert 256*1024*2/2**20 == 0.5
    assert (512//8//2)**2 == 1024 and (1024//8//2)**2 == 4096
    noisy = sqrt(0.64)*2 + sqrt(1-0.64)*(-1)
    reconstructed = (noisy-sqrt(1-0.64)*(-0.5))/sqrt(0.64)
    assert isclose(noisy, 1) and isclose(reconstructed, 1.625)
    assert 50 + 20*40 + 30 == 880


def statistics_examples():
    a, b = [10, 12, 11, 13, 14], [9, 10, 10, 11, 12]
    differences = [x-y for x, y in zip(a, b)]
    standard_error = stdev(differences)/sqrt(len(differences))
    interval = [mean(differences) + sign*2.7764451051977987*standard_error for sign in (-1, 1)]
    # Exhaustive nonparametric resampling of the five paired observations.
    resampled = sorted(mean(differences[i] for i in indices) for indices in product(range(5), repeat=5))
    percentile = [resampled[ceil(p*len(resampled))-1] for p in (0.025, 0.975)]
    assert percentile == [1.2, 2.0]
    failure_upper = 1-0.05**(1/100)
    paired_p = 2*sum(comb(16, k) for k in range(7))/2**16
    assert 0.029 < failure_upper < 0.030
    assert 0.45 < paired_p < 0.46
    assert isclose(mean(a)/mean(b), 15/13)
    print({"mean_difference": mean(differences), "standard_error": standard_error,
           "t_interval": interval, "exact_percentile_bootstrap": percentile,
           "zero_failure_upper": failure_upper, "paired_binary_p": paired_p,
           "twenty_comparisons_false_positive": 1-0.95**20})


if __name__ == "__main__":
    cluster_and_multimodal()
    statistics_examples()
    print("PASS cluster, multimodal, diffusion and statistical teaching models")
