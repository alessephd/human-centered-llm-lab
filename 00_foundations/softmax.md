# Softmax — From First Principles

<img width="375" height="319" alt="image" src="https://github.com/user-attachments/assets/1e5959eb-75ea-4008-bf3c-4c1e5369a285" />


## What is Softmax?

Softmax is a function that transforms a vector of raw scores (logits) into a probability distribution.

Given a vector:

z = [z₁, z₂, ..., zₙ]

Softmax produces:

pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)

Where:
- exp(zᵢ) ensures all outputs are positive
- division by the sum ensures probabilities sum to 1

---

## Why Softmax Matters in Language Models

In a language model:

- The transformer outputs logits (unnormalized scores)
- Softmax converts logits into probabilities over the vocabulary
- The next token is sampled from this probability distribution

Without Softmax:
- We cannot interpret outputs probabilistically
- We cannot compute cross-entropy loss

Softmax is the bridge between raw model output and probabilistic language generation.

---

## Numerical Stability (Critical Detail)

Naive implementation:

exp(z) can overflow when z is large.

Example:
exp(1000) → infinity

To prevent this, we use a numerically stable trick:

z_shifted = z − max(z)

This does NOT change the resulting probabilities.

Why?

Because:

exp(zᵢ − max(z)) / Σ exp(zⱼ − max(z))

The constant cancels out in normalization.

This prevents overflow and stabilizes training.

---

## Geometric Interpretation

Softmax:
- Amplifies differences between logits
- Sharpens dominant values
- Compresses small values

Temperature scaling modifies sharpness:

pᵢ = exp(zᵢ / T) / Σ exp(zⱼ / T)

Where:
- T < 1 → sharper distribution
- T > 1 → smoother distribution

---

## Implementation (Stable Version)

```python
def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

https://www.datacamp.com/pt/tutorial/softmax-activation-function-in-python

https://www.youtube.com/watch?v=ytbYRIN0N4g

