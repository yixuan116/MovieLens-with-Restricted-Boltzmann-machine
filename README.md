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

## RBM Math Walkthrough (Step-by-step)

This section mirrors the implementation in `src/rbm.py` and explains the math flow.

### 1) Model definition

We use a Bernoulli-Bernoulli RBM with visible vector v in {0,1}^V and hidden vector h in {0,1}^H.

Energy:
```
E(v, h) = - v^T W h - b^T v - c^T h
```
where:
- W is V x H weight matrix
- b is visible bias
- c is hidden bias

The joint distribution is:
```
P(v, h) = exp(-E(v, h)) / Z
```
Z is the partition function.

### 2) Conditional probabilities

Because RBM has bipartite structure:
```
P(h_j = 1 | v) = sigmoid( (v W)_j + c_j )
P(v_i = 1 | h) = sigmoid( (h W^T)_i + b_i )
```

These are implemented in:
- `sample_h()` and `sample_v()` in `src/rbm.py`

### 3) Free energy

Marginal over h gives free energy:
```
F(v) = - b^T v - sum_j log(1 + exp((v W)_j + c_j))
```

### 4) Log-likelihood gradient

The gradient for W is:
```
grad_W = E_data[v h^T] - E_model[v h^T]
```
This is the difference between:
- Positive phase: expectation under data distribution
- Negative phase: expectation under model distribution

### 5) Contrastive Divergence (CD-k)

We approximate the negative phase by running a k-step Gibbs chain:

```
v0 -> h0 -> v1 -> h1 -> ... -> vk -> hk
```

Update rules (using probabilities):
```
W += lr * (v0^T h0_prob - vk_prob^T hk_prob) / batch
b += lr * mean(v0 - vk_prob)
c += lr * mean(h0_prob - hk_prob)
```

This corresponds to `contrastive_divergence()` in `src/rbm.py`.

### 6) Reconstruction loss (monitoring)

We track:
```
L = BCE(vk_prob, v0)
```
as a stability/convergence signal (not the exact negative log-likelihood).

## Conceptual Framing

This project emphasizes RBM as a **latent-state discovery engine**, aligned to downstream decision-layer objectives (e.g., regime detection, risk segmentation, policy integration). The same structural inference pattern can be adapted beyond recommendation systems.
