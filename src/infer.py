import argparse
import os

import numpy as np
import pandas as pd
import torch

from data_prep import DataConfig, load_movielens
from rbm import RBM


def main() -> None:
    parser = argparse.ArgumentParser(description="RBM inference for a single user")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--model_path", type=str, default="artifacts/rbm_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    try:
        checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=args.device)

    config = DataConfig(**checkpoint["config"])
    data = load_movielens(config)
    train_matrix = data["train_matrix"]
    user_id_map = checkpoint["user_id_map"]
    item_id_map = checkpoint["item_id_map"]
    id_to_item = {idx: mid for mid, idx in item_id_map.items()}
    movie_titles = load_movie_titles(args.data_dir)

    if args.user_id not in user_id_map:
        raise ValueError(f"user_id {args.user_id} not found after filtering.")

    rbm = RBM(
        n_visible=checkpoint["model_hparams"]["n_visible"],
        n_hidden=checkpoint["model_hparams"]["n_hidden"],
        k=checkpoint["model_hparams"]["k"],
        seed=config.seed,
        device=args.device,
    )
    rbm.load_state_dict(checkpoint["state_dict"])
    rbm.eval()

    user_idx = user_id_map[args.user_id]
    v0 = torch.from_numpy(train_matrix[user_idx].toarray()).float()
    with torch.no_grad():
        scores = rbm.reconstruct(v0).cpu().numpy().ravel()

    seen_items = set(train_matrix[user_idx].indices.tolist())
    scores[list(seen_items)] = -np.inf
    n_items = train_matrix.shape[1]
    k = min(args.topk, n_items)
    topk = np.argpartition(-scores, k - 1)[:k]
    topk = topk[np.argsort(-scores[topk])]
    recommendations = [id_to_item[idx] for idx in topk.tolist()]

    print(f"Top-{k} recommendations for user {args.user_id}:")
    output_rows = []
    for rank, movie_id in enumerate(recommendations, start=1):
        title = movie_titles.get(movie_id, "UNKNOWN_TITLE")
        output_rows.append({"rank": rank, "movieId": int(movie_id), "title": title})
        print(f"{rank:02d}. movieId={movie_id} | {title}")

    if args.save_path:
        save_path = os.path.abspath(args.save_path)
    else:
        artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
        os.makedirs(artifacts_dir, exist_ok=True)
        save_path = os.path.join(artifacts_dir, f"recommendations_user_{args.user_id}.csv")
    pd.DataFrame(output_rows).to_csv(save_path, index=False)
    print(f"Saved recommendations to {save_path}")


def load_movie_titles(data_dir: str) -> dict:
    candidates = ["movies.csv", "movie.csv"]
    movie_path = None
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            movie_path = path
            break
    if movie_path is None:
        return {}
    df = pd.read_csv(movie_path, usecols=["movieId", "title"])
    return dict(zip(df["movieId"].astype(int), df["title"]))


if __name__ == "__main__":
    main()
