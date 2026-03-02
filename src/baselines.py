from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


@dataclass
class PopularityBaseline:
    popularity: Optional[np.ndarray] = None

    def fit(self, train_matrix: sparse.csr_matrix) -> None:
        self.popularity = np.asarray(train_matrix.sum(axis=0)).ravel()

    def predict_scores(self, user_idx: int) -> np.ndarray:
        if self.popularity is None:
            raise ValueError("Popularity baseline is not fitted.")
        return self.popularity


@dataclass
class MatrixFactorizationBaseline:
    n_components: int = 64
    random_state: int = 42
    user_factors: Optional[np.ndarray] = None
    item_factors: Optional[np.ndarray] = None

    def fit(self, train_matrix: sparse.csr_matrix) -> None:
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        user_factors = svd.fit_transform(train_matrix)
        item_factors = svd.components_.T
        self.user_factors = user_factors
        self.item_factors = item_factors

    def predict_scores(self, user_idx: int) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Matrix factorization baseline is not fitted.")
        return self.user_factors[user_idx] @ self.item_factors.T
