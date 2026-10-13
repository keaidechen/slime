"""Recompute book examples using the standard library; no GPU or framework calls.

Run from the repository root:
    python3 docs/ai_infra/examples/rl_infra_contracts.py

Checks use independent formulas, finite differences and explicit negative cases.
They do not establish backend correctness, real throughput or model quality.
"""

import json
import math


def close(actual, expected, tolerance=1e-9):
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError((actual, expected))


def derivative(function, x, step=1e-5):
    return (function(x + step) - function(x - step)) / (2 * step)


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(x - maximum) for x in values))


def blocked_logsumexp(blocks):
    maximum, total = -math.inf, 0.0
    for block in blocks:
        local_max = max(block)
        merged_max = max(maximum, local_max)
        total = total * math.exp(maximum - merged_max) + sum(
            math.exp(x - merged_max) for x in block
        )
        maximum = merged_max
    return maximum + math.log(total)


def probability_and_gradient():
    # Chapters 9/15: independent partitions and extreme finite logits.
    for logits in ([0.0, math.log(2), math.log(3), math.log(4)],
                   [1000.0, 999.0, -1000.0, 998.0]):
        for cut in range(1, len(logits)):
            close(blocked_logsumexp([logits[:cut], logits[cut:]]), logsumexp(logits))
    logits = [math.log(i) for i in (1, 2, 3, 4)]
    log_prob = logits[-1] - blocked_logsumexp([logits[:2], logits[2:]])
    close(math.exp(log_prob), 0.4)
    # A local softmax on the target shard has the wrong denominator.
    local_prob = math.exp(logits[-1] - logsumexp(logits[2:]))
    close(local_prob, 4 / 7)
    for index in range(4):
        def nll(value):
            changed = list(logits)
            changed[index] = value
            return logsumexp(changed) - changed[-1]
        expected = (index + 1) / 10 - int(index == 3)
        close(derivative(nll, logits[index]), expected, tolerance=1e-7)
    # Chapter 5: hold the detached target fixed during finite differences.
    frozen_target = 2.0
    close(derivative(lambda x: x * x, 2.0), 4.0)
    close(derivative(lambda x: x * frozen_target, 2.0), 2.0)
    # Chapter 22: token errors accumulate in the sequence log probability.
    close(math.prod([math.exp(0.01)] * 100), math.exp(1))
    return {"vocab_probability": math.exp(log_prob),
            "wrong_shard_probability": local_prob,
            "sequence_ratio": math.exp(1)}


def preference_and_kl():
    # Chapter 24: logistic loss versus its analytic derivative.
    z = 0.1 * (2 - (-1))
    loss = math.log1p(math.exp(-z))
    close(loss, 0.5543552444685271)
    close(derivative(lambda x: math.log1p(math.exp(-x)), z),
          -1 / (1 + math.exp(z)))
    # Chapter 25: exhaustive expectation, not a noisy Monte Carlo estimate.
    p, q = [0.8, 0.2], [0.5, 0.5]
    k1 = [math.log(a / b) for a, b in zip(p, q)]
    k3 = [b / a - 1 - math.log(b / a) for a, b in zip(p, q)]
    kl = sum(a * value for a, value in zip(p, k1))
    close(kl, sum(a * value for a, value in zip(p, k3)))
    assert min(k1) < 0 and min(k3) >= 0
    # Sampling from q instead of p changes the expectation's meaning.
    wrong_expectation = sum(b * value for b, value in zip(q, k1))
    assert wrong_expectation < 0 < kl
    return {"dpo_loss": loss, "kl_p_q": kl,
            "same_log_ratio_averaged_under_q": wrong_expectation}


def policy_gradient():
    # A two-action policy: probability of action 1 is sigmoid(theta).
    # Reward is [0, 1], independent of theta, so J(theta) = sigmoid(theta).
    def sigmoid(theta):
        return 1 / (1 + math.exp(-theta))
    theta = 0.3
    p = sigmoid(theta)
    probabilities = [1 - p, p]
    rewards = [0, 1]
    score_gradients = [-p, 1 - p]
    for baseline in (0.0, 0.2, 1.0):
        estimate = sum(prob * (reward - baseline) * score for prob, reward, score
                       in zip(probabilities, rewards, score_gradients))
        close(estimate, derivative(sigmoid, theta))
    # Negative case: a one-member group's mean equals its own reward.
    # Centering then produces zero, not the original policy gradient.
    assert p * (1 - p) > 0
    self_centered = sum(prob * (reward - reward) * score for prob, reward, score
                        in zip(probabilities, rewards, score_gradients))
    close(self_centered, 0)
    return {"policy_gradient": p * (1 - p),
            "single_member_centered_gradient": self_centered}


def selection_and_weights():
    # Chapter 26: source distribution times acceptance probability.
    accepted = [0.5 * 0.9, 0.5 * 0.3]
    distribution = [x / sum(accepted) for x in accepted]
    close(distribution[0], 0.75)

    def ess(weights):
        if any(w < 0 for w in weights) or sum(weights) == 0:
            raise ValueError("ESS needs nonnegative weights with positive sum")
        return sum(weights) ** 2 / sum(w * w for w in weights)

    weights = [1, 1, 1, 7]
    clipped = [min(w, 2) for w in weights]
    close(ess([1] * 4), 4)
    close(ess(weights), 100 / 52)
    close(ess(clipped), 25 / 7)
    close(ess([w * 100 for w in weights]), ess(weights))
    try:
        ess([0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("all-zero weights must not produce a valid ESS")
    # Better ESS does not preserve a weighted statistic.
    values = [0, 0, 0, 1]
    before = sum(w * v for w, v in zip(weights, values)) / sum(weights)
    after = sum(w * v for w, v in zip(clipped, values)) / sum(clipped)
    close(before, 0.7)
    close(after, 0.4)
    return {"accepted_task_distribution": distribution, "ess": ess(weights),
            "clipped_ess": ess(clipped), "weighted_value": [before, after]}


def endings_and_queues():
    # Chapter 28: bootstrap and advantage-chain boundaries are independent.
    value, final_value, gamma = 0.4, 0.6, 0.99
    terminal_delta = -value
    truncated_delta = gamma * final_value - value
    close(terminal_delta, -0.4)
    close(truncated_delta, 0.194)
    unrelated_next_advantage = 10.0
    correct_advantage = truncated_delta + gamma * 0.95 * 0 * unrelated_next_advantage
    close(correct_advantage, 0.194)
    wrong_advantage = truncated_delta + gamma * 0.95 * unrelated_next_advantage
    assert wrong_advantage > 9
    # Chapters 21/31: group completion and a fluid queue in one-second slots.
    group_on_time = 0.95 ** 8
    close(group_on_time, 0.6634204312890625)
    queue = 0
    for arrival in [14] * 5:
        queue = max(0, queue + arrival - 10)
    assert queue == 20
    recovery_seconds = 0
    while queue:
        queue = max(0, queue + 8 - 10)
        recovery_seconds += 1
    assert recovery_seconds == 10
    retry_probability = 0.5
    expected_attempts = 1 + retry_probability + retry_probability ** 2
    close(expected_attempts, 1.75)
    return {"terminal_delta": terminal_delta, "truncated_delta": truncated_delta,
            "group_on_time": group_on_time, "queue_recovery_seconds": recovery_seconds,
            "expected_attempts": expected_attempts}


if __name__ == "__main__":
    results = {
        "probability_and_gradient": probability_and_gradient(),
        "preference_and_kl": preference_and_kl(),
        "policy_gradient": policy_gradient(),
        "selection_and_weights": selection_and_weights(),
        "endings_and_queues": endings_and_queues(),
    }
    print(json.dumps(results, indent=2))
    print("PASS book arithmetic and counterexamples; no backend or performance claim")
