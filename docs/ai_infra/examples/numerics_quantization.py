"""Small independent arithmetic checks; no GPU or production quantizer required."""

import math
import struct


def bf16(value):
    """Round finite FP32 input to BF16, ties to even, returned as Python float."""
    bits = struct.unpack("I", struct.pack("f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def quantize(values, scale, zero=0, lower=-7, upper=7):
    codes = [max(lower, min(upper, round(x / scale) + zero)) for x in values]
    return codes, [(q - zero) * scale for q in codes]


def main():
    # Exactly representable operands isolate intermediate rounding.
    left = bf16(bf16(256 + 1) - 256)
    right = bf16(256 + bf16(1 - 256))
    assert (left, right) == (0.0, 1.0)
    assert bf16(1 - 0.001) == 1.0
    master = 1.0
    stored = 1.0
    for _ in range(8):
        master -= 0.001
        stored = bf16(stored - 0.001)
    assert stored == 1.0 and bf16(master) < 1.0

    weights = [[1.0, 0.25], [-0.5, 2.0]]  # Y = X W, columns are output channels.
    x = [2.0, -1.0]
    reference = [sum(x[k] * weights[k][j] for k in range(2)) for j in range(2)]
    assert reference == [2.5, -1.5]
    flat = [v for row in weights for v in row]
    codes, restored = quantize(flat, 2 / 7)
    assert codes == [4, 1, -2, 7]
    tensor_y = [sum(x[k] * restored[2 * k + j] for k in range(2)) for j in range(2)]
    column_w = [[0.0, 0.0], [0.0, 0.0]]
    for j in range(2):
        col = [weights[k][j] for k in range(2)]
        _, decoded = quantize(col, max(abs(v) for v in col) / 7)
        for k in range(2):
            column_w[k][j] = decoded[k]
    column_y = [sum(x[k] * column_w[k][j] for k in range(2)) for j in range(2)]
    assert math.isclose(tensor_y[0], 20 / 7)
    assert math.isclose(column_y[0], 18 / 7)
    assert math.isclose(tensor_y[1], -10 / 7)
    assert math.isclose(column_y[1], -10 / 7)
    # Unsigned asymmetric quantization uses all 16 levels for this range.
    codes_asym, restored_asym = quantize([0, 1, 2, 3], 0.2, lower=0, upper=15)
    assert codes_asym == [0, 5, 10, 15]
    assert restored_asym == [0, 1, 2, 3]
    outliers = [0.1, 0.2, 0.3, 7.0]
    _, full = quantize(outliers, 1.0)
    _, clipped = quantize(outliers, 0.3 / 7)
    assert sum((a - b)**2 for a, b in zip(outliers, full)) < sum(
        (a - b)**2 for a, b in zip(outliers, clipped))
    assert sum((a - b)**2 for a, b in zip(outliers[:3], clipped[:3])) < 0.14

    # Equivalent channel scaling before quantization, not an AWQ implementation.
    scales = [2.0, 0.5]
    scaled_y = [sum((x[k] / scales[k]) * (weights[k][j] * scales[k])
                    for k in range(2)) for j in range(2)]
    assert scaled_y == reference
    # Calibration covariance makes equal weight errors have different costs.
    assert 100 * 0.1**2 > 1 * 0.1**2
    # First-coordinate constraint for inverse Hessian [[2,-1],[-1,2]].
    delta = [-0.2, 0.1]
    assert math.isclose(0.2 + delta[0], 0.0)
    objective = lambda d: (2*d[0]**2 + 2*d[0]*d[1] + 2*d[1]**2) / 3
    assert objective(delta) < objective([-0.2, 0])
    p = [math.exp(v) / sum(math.exp(z) for z in reference) for v in reference]
    q = [math.exp(v) / sum(math.exp(z) for z in tensor_y) for v in tensor_y]
    kl = sum(a * math.log(a / b) for a, b in zip(p, q))
    assert kl >= 0
    print({"BF16_association": [left, right], "reference": reference,
           "tensor_INT4": tensor_y, "channel_INT4": column_y,
           "logprob_delta_action_1": math.log(q[1]) - math.log(p[1]), "KL": kl})
    print("PASS finite-precision, INT4, clipping, scaling and reconstruction checks")


if __name__ == "__main__":
    main()
