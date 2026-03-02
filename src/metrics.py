from typing import Callable, Dict, List, Set

import numpy as np
from scipy import sparse


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def get_user_items(matrix: sparse.csr_matrix, user_idx: int) -> Set[int]:
    row = matrix.getrow(user_idx)
    return set(row.indices.tolist())


def evaluate_model(
    score_fn: Callable[[int], np.ndarray],
    test_matrix: sparse.csr_matrix,
    train_matrix: sparse.csr_matrix,
    k: int,
) -> Dict[str, float]:
    n_users, n_items = train_matrix.shape
    precisions = []
    recalls = []
    ndcgs = []
    recommended_items = set()

    for user_idx in range(n_users):
        relevant = get_user_items(test_matrix, user_idx)
        if not relevant:
            continue
        scores = score_fn(user_idx)
        seen = get_user_items(train_matrix, user_idx)
        scores[list(seen)] = -np.inf
        topk = np.argpartition(-scores, min(k, n_items - 1))[:k]
        topk = topk[np.argsort(-scores[topk])]
        topk_list = topk.tolist()
        recommended_items.update(topk_list)
        precisions.append(precision_at_k(topk_list, relevant, k))
        recalls.append(recall_at_k(topk_list, relevant, k))
        ndcgs.append(ndcg_at_k(topk_list, relevant, k))

    coverage = len(recommended_items) / n_items if n_items > 0 else 0.0
    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": float(coverage),
    }
