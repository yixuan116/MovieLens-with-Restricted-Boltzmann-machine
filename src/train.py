import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines import MatrixFactorizationBaseline, PopularityBaseline
from data_prep import DataConfig, load_movielens, save_config
from metrics import evaluate_model
from rbm import RBM


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_user_indices(n_users: int, batch_size: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = np.arange(n_users)
    rng.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, n_users, batch_size)]


def train_rbm(
    rbm: RBM,
    train_matrix,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> List[float]:
    losses = []
    n_users = train_matrix.shape[0]
    for epoch in range(epochs):
        epoch_losses = []
        for batch in batch_user_indices(n_users, batch_size, seed + epoch):
            v0 = torch.from_numpy(train_matrix[batch].toarray()).float()
            loss = rbm.contrastive_divergence(v0, lr=lr)
            epoch_losses.append(loss)
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {losses[-1]:.6f}")
    return losses


def plot_losses(losses: List[float], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title("RBM Training Loss")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_rbm(rbm: RBM, train_matrix, test_matrix, k: int, batch_size: int) -> Dict[str, float]:
    n_users = train_matrix.shape[0]
    n_items = train_matrix.shape[1]

    def score_fn(user_idx: int) -> np.ndarray:
        v0 = torch.from_numpy(train_matrix[user_idx].toarray()).float()
        with torch.no_grad():
            scores = rbm.reconstruct(v0).cpu().numpy().ravel()
        return scores

    return evaluate_model(score_fn, test_matrix, train_matrix, k)


def evaluate_baselines(
    train_matrix,
    test_matrix,
    k: int,
    svd_components: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    popularity = PopularityBaseline()
    popularity.fit(train_matrix)
    pop_metrics = evaluate_model(popularity.predict_scores, test_matrix, train_matrix, k)

    mf = MatrixFactorizationBaseline(n_components=svd_components, random_state=seed)
    mf.fit(train_matrix)
    mf_metrics = evaluate_model(mf.predict_scores, test_matrix, train_matrix, k)

    return {"popularity": pop_metrics, "matrix_factorization": mf_metrics}


def save_metrics(metrics: Dict[str, Dict[str, float]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RBM for MovieLens 20M")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_user_interactions", type=int, default=5)
    parser.add_argument("--min_item_interactions", type=int, default=5)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--svd_components", type=int, default=64)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    config = DataConfig(
        data_dir=args.data_dir,
        min_rating=args.min_rating,
        test_ratio=args.test_ratio,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_users=args.max_users,
        max_items=args.max_items,
        seed=args.seed,
    )
    data = load_movielens(config)
    train_matrix = data["train_matrix"]
    test_matrix = data["test_matrix"]

    rbm = RBM(
        n_visible=train_matrix.shape[1],
        n_hidden=args.hidden_units,
        k=args.k,
        seed=args.seed,
        device=args.device,
    )

    losses = train_rbm(
        rbm=rbm,
        train_matrix=train_matrix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    artifacts_dir = os.path.abspath(artifacts_dir)
    metrics_dir = os.path.join(artifacts_dir, "metrics")
    figures_dir = os.path.join(artifacts_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    plot_losses(losses, os.path.join(figures_dir, "rbm_loss_curve.png"))
    rbm_metrics = evaluate_rbm(rbm, train_matrix, test_matrix, args.topk, args.batch_size)
    baselines_metrics = evaluate_baselines(
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        k=args.topk,
        svd_components=args.svd_components,
        seed=args.seed,
    )

    all_metrics = {"rbm": rbm_metrics, **baselines_metrics}
    save_metrics(all_metrics, os.path.join(metrics_dir, "model_comparison.json"))

    summary = {
        "n_users": int(train_matrix.shape[0]),
        "n_items": int(train_matrix.shape[1]),
        "hidden_units": args.hidden_units,
        "k": args.k,
        "epochs": args.epochs,
        "loss_curve": losses,
    }
    save_metrics(summary, os.path.join(metrics_dir, "latent_summary.json"))

    checkpoint = {
        "state_dict": rbm.state_dict(),
        "config": config.__dict__,
        "model_hparams": {
            "n_visible": train_matrix.shape[1],
            "n_hidden": args.hidden_units,
            "k": args.k,
        },
        "user_id_map": data["user_id_map"],
        "item_id_map": data["item_id_map"],
    }
    torch.save(checkpoint, os.path.join(artifacts_dir, "rbm_model.pt"))
    save_config(config, os.path.join(metrics_dir, "data_config.json"))

    print("Training complete. Metrics saved to artifacts/metrics.")


if __name__ == "__main__":
    main()
