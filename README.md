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
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}E(v,h)=-v%5ETWh-b%5ETv-c%5ETh" alt="E(v,h) = - v^T W h - b^T v - c^T h" />
</p>
where:
- W is V x H weight matrix
- b is visible bias
- c is hidden bias

The joint distribution is:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}P(v,h)=%5Cfrac%7B%5Cexp(-E(v,h))%7D%7BZ%7D" alt="P(v,h) = exp(-E(v,h)) / Z" />
</p>
Z is the partition function.

### 2) Conditional probabilities

Because RBM has bipartite structure:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}P(h_j=1%5Cmid%20v)=%5Csigma((vW)_j%2Bc_j)" alt="P(h_j=1|v) = sigma((v W)_j + c_j)" />
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}P(v_i=1%5Cmid%20h)=%5Csigma((hW%5ET)_i%2Bb_i)" alt="P(v_i=1|h) = sigma((h W^T)_i + b_i)" />
</p>

These are implemented in:
- `sample_h()` and `sample_v()` in `src/rbm.py`

### 3) Free energy

Marginal over h gives free energy:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}F(v)=-b%5ETv-%5Csum_j%5Clog(1%2B%5Cexp((vW)_j%2Bc_j))" alt="F(v) = - b^T v - sum_j log(1 + exp((v W)_j + c_j))" />
</p>

### 4) Log-likelihood gradient

The gradient for W is:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}%5Cnabla_W=%5Cmathbb%7BE%7D_%7Bdata%7D%5Bvh%5ET%5D-%5Cmathbb%7BE%7D_%7Bmodel%7D%5Bvh%5ET%5D" alt="nabla_W = E_data[v h^T] - E_model[v h^T]" />
</p>
This is the difference between:
- Positive phase: expectation under data distribution
- Negative phase: expectation under model distribution

### 5) Contrastive Divergence (CD-k)

We approximate the negative phase by running a k-step Gibbs chain:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}v_0%5Crightarrow%20h_0%5Crightarrow%20v_1%5Crightarrow%20h_1%5Crightarrow%20%5Ccdots%5Crightarrow%20v_k%5Crightarrow%20h_k" alt="v0 -> h0 -> v1 -> h1 -> ... -> vk -> hk" />
</p>

Update rules (using probabilities):
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}W%5Cleftarrow%20W%2B%5Ceta%5Cfrac%7Bv_0%5ETh_0%5E%7Bprob%7D-(v_k%5E%7Bprob%7D)%5ETh_k%5E%7Bprob%7D%7D%7Bbatch%7D" alt="W <- W + eta * (v0^T h0_prob - vk_prob^T hk_prob) / batch" />
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}b%5Cleftarrow%20b%2B%5Ceta%5C,%20mean(v_0-v_k%5E%7Bprob%7D)" alt="b <- b + eta * mean(v0 - vk_prob)" />
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}c%5Cleftarrow%20c%2B%5Ceta%5C,%20mean(h_0%5E%7Bprob%7D-h_k%5E%7Bprob%7D)" alt="c <- c + eta * mean(h0_prob - hk_prob)" />
</p>

This corresponds to `contrastive_divergence()` in `src/rbm.py`.

### 6) Reconstruction loss (monitoring)

We track:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cdpi{120}L=%5Cmathrm%7BBCE%7D(v_k%5E%7Bprob%7D,v_0)" alt="L = BCE(vk_prob, v0)" />
</p>
as a stability/convergence signal (not the exact negative log-likelihood).

## Conceptual Framing

This project emphasizes RBM as a **latent-state discovery engine**, aligned to downstream decision-layer objectives (e.g., regime detection, risk segmentation, policy integration). The same structural inference pattern can be adapted beyond recommendation systems.
