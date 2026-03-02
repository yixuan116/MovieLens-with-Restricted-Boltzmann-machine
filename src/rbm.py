from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    def __init__(self, n_visible: int, n_hidden: int, k: int = 1, seed: int = 42, device: str = "cpu"):
        super().__init__()
        torch.manual_seed(seed)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.device = torch.device(device)

        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

        self.to(self.device)

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """p(h=1|v) = sigmoid(v W + h_bias), then Bernoulli sample."""
        prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        sample = torch.bernoulli(prob)
        return prob, sample

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """p(v=1|h) = sigmoid(h W^T + v_bias), then Bernoulli sample."""
        prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        sample = torch.bernoulli(prob)
        return prob, sample

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """F(v) = -v^T v_bias - sum_j log(1 + exp((v W)_j + h_bias_j))."""
        vbias_term = torch.matmul(v, self.v_bias)
        hidden_term = torch.sum(F.softplus(torch.matmul(v, self.W) + self.h_bias), dim=1)
        return -vbias_term - hidden_term

    def gibbs_sampling(self, v0: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run k-step Gibbs chain: v -> h -> v -> ... returning v_k prob and h_k prob."""
        vk = v0
        hk_prob = None
        for _ in range(k):
            hk_prob, hk_sample = self.sample_h(vk)
            vk_prob, vk = self.sample_v(hk_sample)
        return vk_prob, hk_prob

    def contrastive_divergence(self, v0: torch.Tensor, lr: float) -> float:
        """CD-k update using positive phase (v0,h0) and negative phase (v_k,h_k)."""
        v0 = v0.to(self.device)
        h0_prob, h0_sample = self.sample_h(v0)
        vk_prob, hk_prob = self.gibbs_sampling(v0, self.k)

        batch_size = v0.size(0)
        pos_grad = torch.matmul(v0.t(), h0_prob)
        neg_grad = torch.matmul(vk_prob.t(), hk_prob)

        self.W.data += lr * (pos_grad - neg_grad) / batch_size
        self.v_bias.data += lr * torch.mean(v0 - vk_prob, dim=0)
        self.h_bias.data += lr * torch.mean(h0_prob - hk_prob, dim=0)

        loss = F.binary_cross_entropy(vk_prob, v0, reduction="mean")
        return float(loss.item())

    @torch.no_grad()
    def reconstruct(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(self.device)
        h_prob, _ = self.sample_h(v)
        v_prob, _ = self.sample_v(h_prob)
        return v_prob
