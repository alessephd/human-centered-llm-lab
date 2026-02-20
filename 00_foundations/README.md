# 00_foundations

This module contains first-principles implementations of core mathematical building blocks required for training transformer-based language models.

## Why this matters

Before building large systems, we must understand:

- Softmax behavior
- Cross-entropy loss
- Gradient flow
- Numerical stability
- Tensor shapes

If I cannot derive and implement it from scratch,
I do not truly understand it.

## Implementations in this module

- softmax.py
- cross_entropy.py
- tests/

## Current Status
✔ Softmax implemented  
✔ Cross-entropy implemented  
⬜ Numerical stability analysis  
⬜ Gradient verification  

## Next Step
Compare manual implementation with PyTorch’s native functions.
