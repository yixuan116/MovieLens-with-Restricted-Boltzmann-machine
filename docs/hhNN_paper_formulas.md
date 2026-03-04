# Hyperbolic Hopfield Neural Networks (Kobayashi, 2013) — Key Formulas

Paper: *Hyperbolic Hopfield Neural Networks* (IEEE TNNLS, 2013).

This note lists the main equations used in the paper. The model is **hyperbolic Hopfield neural networks (HHNNs)**, which is different from RBM, but shares the idea of energy-based stability.

## Section II — Complex-Valued Hopfield Neural Networks (CHNNs)

**Activation (complex neuron):**
```math
f(z) = \frac{z}{|z|} = \frac{x + i y}{\sqrt{x^2 + y^2}}
```

**Symmetry condition (weights):**
```math
w_{kj} = w_{jk}
```

**Weighted sum input:**
```math
I_k = \sum_{j \ne k} w_{kj} z_j
```

**Energy (CHNN):**
```math
E = -\frac{1}{2} \sum_k \sum_{j \ne k} z_k w_{kj} z_j
```

**Complex Hebbian rule:**
```math
w_{kj} = \sum_p c_k^p c_j^p
```

## Section III — Hyperbolic Numbers

**Hyperbolic number:**
```math
z = x + u y, \quad u^2 = 1
```

**Addition / multiplication:**
```math
(x + u y) + (x' + u y') = (x + x') + u (y + y')
```
```math
(x + u y)(x' + u y') = (x x' + y y') + u(x y' + x' y)
```

**Conjugate and modulus:**
```math
\bar{z} = x - u y
```
```math
|z| = \sqrt{|z \bar{z}|} = \sqrt{|x^2 - y^2|}
```

**Domain and unit hyperbola:**
```math
D = \{ z = x + u y \in \mathbb{H} : x > 0,\ x^2 > y^2 \}
```
```math
S = \{ z \in D : |z| = 1 \}
```

**Hyperbolic exponential:**
```math
\exp(u t) = \cosh t + u \sinh t
```

**Hyperbolic polar form:**
```math
z = r \exp(u t),\quad r = |z| = \sqrt{x^2 - y^2},\quad t = \tanh^{-1}\frac{y}{x}
```

**Multiplication in polar form:**
```math
z_1 z_2 = r_1 r_2 \exp(u (t_1 + t_2))
```

## Section IV — HHNN Model

**Hyperbolic activation:**
```math
g(z) = \frac{z}{|z|} = \frac{x + u y}{\sqrt{x^2 - y^2}}
```

**HHNN weight constraints:**
```math
w_{kj} \in D,\quad w_{kj} = w_{jk}
```

**Weighted sum input:**
```math
I_k = \sum_{j \ne k} w_{kj} z_j
```

**Energy (HHNN):**
```math
E = \frac{1}{2} \sum_k \sum_{j \ne k} z_k w_{kj} z_j
```

**Energy gap under update:**
```math
\Delta E = (z_l' I_l)_r - (z_l I_l)_r
```
Updating with the hyperbolic activation minimizes this term, so energy does not increase.

**Hyperbolic Hebbian rule:**
```math
w_{kj} = \sum_p c_k^p c_j^p
```

## Notes

- The paper highlights **hyperbolic rotation invariance** and the fact that hyperbolic neuron states lie on a hyperbola rather than a circle.
- The HHNN is energy-based but **not an RBM**. RBM training uses contrastive divergence, while HHNN uses Hebbian-style rules and energy minimization.
