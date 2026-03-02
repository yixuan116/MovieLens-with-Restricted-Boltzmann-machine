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

## RBM Math Walkthrough (Full Derivation)

This section mirrors the implementation in `src/rbm.py` and expands the math flow into a complete derivation.

### 1) Model definition and joint distribution

We use a Bernoulli-Bernoulli RBM with visible vector v in {0,1}^V and hidden vector h in {0,1}^H.

Energy:

$$
E(v, h) = - v^T W h - b^T v - c^T h
$$
where:
- W is V x H weight matrix
- b is visible bias
- c is hidden bias

The joint distribution is:

$$
P(v, h) = \frac{\exp(-E(v, h))}{Z}
$$
Z is the partition function, and the marginals are:

$$
P(v) = \sum_h P(v, h), \quad P(h) = \sum_v P(v, h)
$$

### 2) Conditional probabilities

Because RBM has bipartite structure:

$$
P(h_j = 1 \mid v) = \sigma((v W)_j + c_j)
$$

$$
P(v_i = 1 \mid h) = \sigma((h W^T)_i + b_i)
$$

Derivation sketch for the hidden conditional:

$$
P(h_j = 1 \mid v) = \frac{\exp(v^T W_{:,j} + c_j)}{1 + \exp(v^T W_{:,j} + c_j)}
$$

These are implemented in:
- `sample_h()` and `sample_v()` in `src/rbm.py`

### 3) Free energy

Marginal over h gives free energy:

$$
F(v) = - b^T v - \sum_j \log(1 + \exp((v W)_j + c_j))
$$

Derivation:

$$
P(v) = \frac{1}{Z} \sum_h \exp(-E(v,h))
$$

$$
\sum_h \exp(-E(v,h)) = \exp(b^T v) \prod_j (1 + \exp((vW)_j + c_j))
$$
Therefore:

$$
P(v) = \frac{\exp(-F(v))}{Z}
$$

### 4) Log-likelihood gradient

We maximize log-likelihood:

$$
\mathcal{L} = \sum_{v \in \text{data}} \log P(v)
$$
For one data vector v:

$$
\nabla_W \log P(v) = -\nabla_W F(v) - \nabla_W \log Z
$$
The first term is the positive phase:

$$
-\nabla_W F(v) = v \, \mathbb{E}_{P(h \mid v)}[h]^T
$$
The second term is the negative phase:

$$
\nabla_W \log Z = \mathbb{E}_{P(v,h)}[v h^T]
$$

So the gradient for W is:

$$
\nabla_W = \mathbb{E}_{data}[v h^T] - \mathbb{E}_{model}[v h^T]
$$
This is the difference between:
- Positive phase: expectation under data distribution
- Negative phase: expectation under model distribution

Similarly for biases:

$$
\nabla_b = \mathbb{E}_{data}[v] - \mathbb{E}_{model}[v]
$$

$$
\nabla_c = \mathbb{E}_{data}[h] - \mathbb{E}_{model}[h]
$$

### 5) Contrastive Divergence (CD-k)

We approximate the negative phase by running a k-step Gibbs chain:


$$
v_0 \rightarrow h_0 \rightarrow v_1 \rightarrow h_1 \rightarrow \cdots \rightarrow v_k \rightarrow h_k
$$

Update rules (using probabilities):

$$
W \leftarrow W + \eta \frac{v_0^T h_0^{prob} - (v_k^{prob})^T h_k^{prob}}{batch}
$$

$$
b \leftarrow b + \eta \, mean(v_0 - v_k^{prob})
$$

$$
c \leftarrow c + \eta \, mean(h_0^{prob} - h_k^{prob})
$$

Interpretation:
- v_0, h_0 are from the data-driven positive phase
- v_k, h_k are from the model-driven negative phase after k Gibbs steps
- k is small (e.g., 1, 5) for efficient approximation

This corresponds to `contrastive_divergence()` in `src/rbm.py`.

### 6) Reconstruction loss (monitoring)

We track:

$$
L = BCE(v_k^{prob}, v_0)
$$
as a stability/convergence signal (not the exact negative log-likelihood).

## Conceptual Framing

This project emphasizes RBM as a **latent-state discovery engine**, aligned to downstream decision-layer objectives (e.g., regime detection, risk segmentation, policy integration). The same structural inference pattern can be adapted beyond recommendation systems.
