import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class DataConfig:
    data_dir: str
    min_rating: float = 4.0
    test_ratio: float = 0.2
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    max_users: Optional[int] = None
    max_items: Optional[int] = None
    seed: int = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def load_ratings_csv(data_dir: str) -> pd.DataFrame:
    candidates = ["ratings.csv", "rating.csv"]
    ratings_path = None
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            ratings_path = path
            break
    if ratings_path is None:
        raise FileNotFoundError(
            f"No ratings file found in {data_dir}. "
            f"Expected one of: {', '.join(candidates)}"
        )
    df = pd.read_csv(ratings_path)
    return df[["userId", "movieId", "rating", "timestamp"]]


def binarize_ratings(df: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    df = df.copy()
    df["label"] = (df["rating"] >= min_rating).astype(np.int8)
    return df[df["label"] == 1][["userId", "movieId", "timestamp"]]


def filter_min_interactions(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    user_counts = df["userId"].value_counts()
    item_counts = df["movieId"].value_counts()
    df = df[df["userId"].isin(user_counts[user_counts >= min_user_interactions].index)]
    df = df[df["movieId"].isin(item_counts[item_counts >= min_item_interactions].index)]
    return df


def downsample_users_items(
    df: pd.DataFrame,
    max_users: Optional[int],
    max_items: Optional[int],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    if max_users is not None:
        user_ids = df["userId"].unique()
        if len(user_ids) > max_users:
            keep_users = rng.choice(user_ids, size=max_users, replace=False)
            df = df[df["userId"].isin(keep_users)]
    if max_items is not None:
        item_ids = df["movieId"].unique()
        if len(item_ids) > max_items:
            keep_items = rng.choice(item_ids, size=max_items, replace=False)
            df = df[df["movieId"].isin(keep_items)]
    return df


def time_aware_split(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["userId", "timestamp"])
    train_parts = []
    test_parts = []
    for _, group in df.groupby("userId", sort=False):
        n = len(group)
        if n <= 1:
            train_parts.append(group)
            continue
        test_size = max(1, int(np.ceil(n * test_ratio)))
        split_idx = max(1, n - test_size)
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0]
    return train_df, test_df


def build_mappings(train_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    user_ids = np.sort(train_df["userId"].unique())
    item_ids = np.sort(train_df["movieId"].unique())
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {mid: idx for idx, mid in enumerate(item_ids)}
    return user_id_map, item_id_map


def build_matrix(
    df: pd.DataFrame,
    user_id_map: Dict[int, int],
    item_id_map: Dict[int, int],
) -> sparse.csr_matrix:
    user_indices = df["userId"].map(user_id_map).dropna().astype(int)
    item_indices = df["movieId"].map(item_id_map).dropna().astype(int)
    data = np.ones(len(user_indices), dtype=np.float32)
    mat = sparse.csr_matrix(
        (data, (user_indices.values, item_indices.values)),
        shape=(len(user_id_map), len(item_id_map)),
        dtype=np.float32,
    )
    mat.eliminate_zeros()
    return mat


def load_movielens(config: DataConfig) -> Dict[str, object]:
    set_seed(config.seed)
    raw_df = load_ratings_csv(config.data_dir)
    df = binarize_ratings(raw_df, config.min_rating)
    df = filter_min_interactions(
        df,
        min_user_interactions=config.min_user_interactions,
        min_item_interactions=config.min_item_interactions,
    )
    df = downsample_users_items(
        df,
        max_users=config.max_users,
        max_items=config.max_items,
        seed=config.seed,
    )
    train_df, test_df = time_aware_split(df, config.test_ratio)
    user_id_map, item_id_map = build_mappings(train_df)
    test_df = test_df[
        test_df["userId"].isin(user_id_map)
        & test_df["movieId"].isin(item_id_map)
    ]
    train_matrix = build_matrix(train_df, user_id_map, item_id_map)
    test_matrix = build_matrix(test_df, user_id_map, item_id_map)
    return {
        "train_matrix": train_matrix,
        "test_matrix": test_matrix,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "train_df": train_df,
        "test_df": test_df,
    }


def save_config(config: DataConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2, sort_keys=True)
