"""A complete two-token Transformer with scalar reverse-mode autodiff.

Python standard library only. FP64 teaching arithmetic, not a GPU kernel.
"""

import math


class Value:
    def __init__(self, data, parents=()):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents  # (parent, local derivative) pairs

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, ((self, 1.0), (other, 1.0)))

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, ((self, other.data), (other, self.data)))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1)

    def __pow__(self, power):
        return Value(self.data**power, ((self, power * self.data**(power - 1)),))

    def exp(self):
        out = math.exp(self.data)
        return Value(out, ((self, out),))

    def log(self):
        return Value(math.log(self.data), ((self, 1 / self.data),))

    def backward(self):
        order, visited = [], set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent, _ in node.parents:
                visit(parent)
            order.append(node)

        visit(self)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            for parent, derivative in node.parents:
                parent.grad += node.grad * derivative


def initial_parameters():
    return {
        "embedding": [[0.5, -0.25], [0.25, 0.75], [-0.5, 0.5]],
        "q": [[0.5, 0.25], [-0.25, 0.5]],
        "k": [[0.25, -0.5], [0.5, 0.25]],
        "v": [[0.5, 0.0], [0.25, 0.5]],
        "o": [[0.5, -0.25], [0.25, 0.5]],
        "up": [[0.25, 0.5], [-0.5, 0.25]],
        "down": [[0.5, 0.25], [-0.25, 0.5]],
        "head": [[0.5, -0.25, 0.25], [-0.25, 0.5, 0.0]],
    }


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def rmsnorm(rows):
    return [[x * (sum(v*v for v in row) / len(row) + 1e-5)**-0.5 for x in row]
            for row in rows]


def forward(parameters, tokens=(0, 1), labels=(1, 2)):
    if len(tokens) != len(labels) or not tokens:
        raise ValueError("Each input position needs one next-token label")
    p = {name: [[Value(v) for v in row] for row in matrix]
         for name, matrix in parameters.items()}
    # Fixed positions are constants; causal mask alone does not encode position.
    x = [[v + 0.1 * position * (1 if j == 0 else -1)
          for j, v in enumerate(p["embedding"][token])]
         for position, token in enumerate(tokens)]
    norm = rmsnorm(x)
    q, k, v = (matmul(norm, p[name]) for name in ("q", "k", "v"))
    attention, probabilities = [], []
    for i, query in enumerate(q):
        scores = [sum(a*b for a, b in zip(query, key)) / math.sqrt(2) for key in k[:i+1]]
        maximum = max(z.data for z in scores)  # constant shift cancels in softmax
        exponentials = [(z - maximum).exp() for z in scores]
        probs = [e / sum(exponentials) for e in exponentials]
        probabilities.append([z.data for z in probs])
        attention.append([sum(probs[j] * v[j][d] for j in range(i+1)) for d in range(2)])
    projected = matmul(attention, p["o"])
    residual = [[a+b for a, b in zip(rx, ro)] for rx, ro in zip(x, projected)]
    up = matmul(rmsnorm(residual), p["up"])
    hidden = [[z / (1 + (-z).exp()) for z in row] for row in up]
    down = matmul(hidden, p["down"])
    output = [[a+b for a, b in zip(ru, rd)] for ru, rd in zip(residual, down)]
    logits = matmul(rmsnorm(output), p["head"])
    losses = []
    for row, label in zip(logits, labels):
        maximum = max(z.data for z in row)
        losses.append(sum((z-maximum).exp() for z in row).log() + maximum - row[label])
    loss = sum(losses) / len(losses)
    states = {"input": [[z.data for z in row] for row in x],
              "attention_probabilities": probabilities,
              "logits": [[z.data for z in row] for row in logits],
              "token_losses": [z.data for z in losses]}
    return loss, p, states


def gradients(parameters, tokens=(0, 1), labels=(1, 2)):
    loss, variables, states = forward(parameters, tokens, labels)
    loss.backward()
    grads = {name: [[v.grad for v in row] for row in matrix] for name, matrix in variables.items()}
    return loss.data, grads, states


def main():
    parameters = initial_parameters()
    loss, grads, states = gradients(parameters)
    largest = 0.0
    count = 0
    for name, matrix in parameters.items():
        for i, row in enumerate(matrix):
            for j, original in enumerate(row):
                row[j] = original + 1e-5
                plus = forward(parameters)[0].data
                row[j] = original - 1e-5
                minus = forward(parameters)[0].data
                row[j] = original
                numerical = (plus - minus) / 2e-5
                error = abs(numerical - grads[name][i][j])
                largest = max(largest, error)
                assert error < 2e-6, (name, i, j, error)
                count += 1
    # Changing the future token must not change the first-position logits.
    other = forward(parameters, (0, 2))[2]
    assert states["logits"][0] == other["logits"][0]
    updated = {name: [[v - 0.01*g for v, g in zip(row, grow)]
                       for row, grow in zip(matrix, grads[name])]
               for name, matrix in parameters.items()}
    assert forward(updated)[0].data < loss
    print({"loss": loss, "states": states, "head_gradient": grads["head"],
           "parameters_checked": count, "max_gradient_error": largest,
           "loss_after_SGD": forward(updated)[0].data})
    print("PASS complete forward/backward, finite differences and causal boundary")


if __name__ == "__main__":
    main()
