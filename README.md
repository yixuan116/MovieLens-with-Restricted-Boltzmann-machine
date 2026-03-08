# RBM RecSys (MovieLens 20M)

This repository implements a modular, production-style Restricted Boltzmann Machine (RBM) recommender in PyTorch with strong engineering hygiene: reproducibility, baselines, structured evaluation, and artifact outputs.

The RBM is treated as a latent-structure inference layer. The pipeline emphasizes structural decomposition, stability analysis via convergence curves, and compatibility with decision-layer integration.

## Project Structure

```
rbm-recsys/
├── src/
│   ├── data_prep.py
│   ├── rbm.py
│   ├── baselines.py
│   ├── metrics.py
│   ├── train.py
│   └── infer.py
├── notebooks/
├── artifacts/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

## Dataset

MovieLens 20M (Kaggle):
```
kaggle datasets download -d grouplens/movielens-20m-dataset -p data --unzip
```

The dataset must **not** be committed to the repository.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```
python src/train.py \
  --data_dir data/movielens-20m \
  --hidden_units 128 \
  --k 5 \
  --epochs 20
```

For large-scale runs, consider sampling:
```
python src/train.py \
  --data_dir data/movielens-20m \
  --max_users 5000 \
  --max_items 8000 \
  --epochs 10
```

## Inference

```
python src/infer.py \
  --data_dir data/movielens-20m \
  --user_id 123 \
  --topk 20
```

## Outputs

Artifacts are saved under `artifacts/`:

- `figures/rbm_loss_curve.png`
- `metrics/model_comparison.json`
- `metrics/latent_summary.json`
- `rbm_model.pt`

## Model Pipeline and Training Architecture

This section explains the end-to-end system architecture used in this project:
MovieLens dataset → RBM recommender → Hyperbolic neural dynamics.

### 1. Data Representation

The MovieLens dataset is structured as user ratings with timestamps. It is converted into a user–movie interaction matrix.

Shape: 2000 × 1320

Rows = users  
Columns = movies

Each row represents a single user's interaction vector and becomes one training sample.

Important clarification:
The 2000×1320 matrix is NOT a neural network layer.
It is the training dataset.

The RBM visible layer size equals the number of movies (1320).

### 2. System Architecture

MovieLens Dataset  
(rating.csv)

↓

User–Movie Interaction Matrix  
2000 × 1320

↓

RBM Visible Layer  
v ∈ {0,1}^{1320}

(each node represents one movie)

↓

Weight Matrix  
W ∈ ℝ^{1320 × K}

↓

Hidden Layer  
h ∈ {0,1}^K

(latent user preference factors)

↓

RBM Reconstruction  
v̂ ∈ ℝ^{1320}

(predicted preference probabilities)

↓

Hyperbolic Mapping

z = cosh(t₀) ± u sinh(t₀)

↓

Hyperbolic Weighted Sum

Iₖ = Σ wₖⱼ zⱼ

↓

Activation g(z)

↓

Stable / Equilibrium State

The RBM learns latent user preference factors from the visible movie ratings.
The hidden layer captures abstract taste features.
The reconstruction step predicts missing movie preferences.

The hyperbolic mapping converts binary states into hyperbolic-valued states.
The hyperbolic network dynamics compute stable equilibrium states.

### 3. RBM Training Loop

Input vector (one user)

v

↓

Hidden activation

h = σ(Wᵀv + b)

↓

Reconstruction

v̂ = σ(Wh + a)

↓

Contrastive Divergence

ΔW = v hᵀ − v̂ ĥᵀ

↓

Update weights

W ← W + ηΔW

RBM training alternates between visible and hidden layers.
The model learns weights that minimize reconstruction error.

This process discovers latent preference structures in the movie space.

### 4. Energy Landscape Intuition

Real-valued energy landscape

  ~ oscillatory surface

multiple unstable regions

Hyperbolic energy landscape

  \        /
   \      /
    \____/

stable attractor basin

Traditional real-valued networks can exhibit oscillatory dynamics.
Hyperbolic-valued states introduce exponential geometry that promotes convergence.

The hyperbolic energy landscape forms stable attractor basins.
This helps the system reach equilibrium states more reliably.

### Final Summary

The recommendation system pipeline is:

MovieLens Data  
→ RBM latent preference learning  
→ Hyperbolic state transformation  
→ Stable equilibrium recommendation

## RBM Math Walkthrough (Full Derivation)

This section mirrors the implementation in `src/rbm.py` and expands the math flow into a complete derivation.

### 1) Model definition and joint distribution

We use a Bernoulli-Bernoulli RBM with visible vector v in {0,1}^V and hidden vector h in {0,1}^H.

Energy:

```math
E(v, h) = - v^T W h - b^T v - c^T h
```
where:
- W is V x H weight matrix
- b is visible bias
- c is hidden bias

The joint distribution is:

```math
P(v, h) = \frac{\exp(-E(v, h))}{Z}
```
Z is the partition function, and the marginals are:

```math
P(v) = \sum_h P(v, h), \quad P(h) = \sum_v P(v, h)
```

### 2) Conditional probabilities

Because RBM has bipartite structure:

```math
P(h_j = 1 \mid v) = \sigma((v W)_j + c_j)
```

```math
P(v_i = 1 \mid h) = \sigma((h W^T)_i + b_i)
```

Derivation sketch for the hidden conditional:

```math
P(h_j = 1 \mid v) = \frac{\exp(v^T W_{:,j} + c_j)}{1 + \exp(v^T W_{:,j} + c_j)}
```

These are implemented in:
- `sample_h()` and `sample_v()` in `src/rbm.py`

### 3) Free energy

Marginal over h gives free energy:

```math
F(v) = - b^T v - \sum_j \log(1 + \exp((v W)_j + c_j))
```

Derivation:

```math
P(v) = \frac{1}{Z} \sum_h \exp(-E(v,h))
```

```math
\sum_h \exp(-E(v,h)) = \exp(b^T v) \prod_j (1 + \exp((vW)_j + c_j))
```
Therefore:

```math
P(v) = \frac{\exp(-F(v))}{Z}
```

### 4) Log-likelihood gradient

We maximize log-likelihood:

```math
\mathcal{L} = \sum_{v \in \text{data}} \log P(v)
```
For one data vector v:

```math
\nabla_W \log P(v) = -\nabla_W F(v) - \nabla_W \log Z
```
The first term is the positive phase:

```math
-\nabla_W F(v) = v \, \mathbb{E}_{P(h \mid v)}[h]^T
```
The second term is the negative phase:

```math
\nabla_W \log Z = \mathbb{E}_{P(v,h)}[v h^T]
```

So the gradient for W is:

```math
\nabla_W = \mathbb{E}_{data}[v h^T] - \mathbb{E}_{model}[v h^T]
```
This is the difference between:
- Positive phase: expectation under data distribution
- Negative phase: expectation under model distribution

Similarly for biases:

```math
\nabla_b = \mathbb{E}_{data}[v] - \mathbb{E}_{model}[v]
```

```math
\nabla_c = \mathbb{E}_{data}[h] - \mathbb{E}_{model}[h]
```

### 5) Contrastive Divergence (CD-k)

We approximate the negative phase by running a k-step Gibbs chain:

```math
v_0 \rightarrow h_0 \rightarrow v_1 \rightarrow h_1 \rightarrow \cdots \rightarrow v_k \rightarrow h_k
```

Update rules (using probabilities):

```math
W \leftarrow W + \eta \frac{v_0^T h_0^{prob} - (v_k^{prob})^T h_k^{prob}}{batch}
```

```math
b \leftarrow b + \eta \, mean(v_0 - v_k^{prob})
```

```math
c \leftarrow c + \eta \, mean(h_0^{prob} - h_k^{prob})
```

Interpretation:
- v_0, h_0 are from the data-driven positive phase
- v_k, h_k are from the model-driven negative phase after k Gibbs steps
- k is small (e.g., 1, 5) for efficient approximation

This corresponds to `contrastive_divergence()` in `src/rbm.py`.

### 6) Reconstruction loss (monitoring)

We track:

```math
L = BCE(v_k^{prob}, v_0)
```
as a stability/convergence signal (not the exact negative log-likelihood).

## Conceptual Framing

This project emphasizes RBM as a **latent-state discovery engine**, aligned to downstream decision-layer objectives (e.g., regime detection, risk segmentation, policy integration). The same structural inference pattern can be adapted beyond recommendation systems.
