"""Channel-selective CD-1 update demo for user 3268.

Uses the already-trained e+ (channel 1, rating) and e- (channel 2,
personality) RBM weights -- no retraining. Runs one CD-1 step for this one
user, decomposes the resulting energy change hidden-unit by hidden-unit into
Delta1 (e+ channel) and Delta2 (e- channel), and compares two update-accept
strategies:

  - Real RBM: a single scalar decision per hidden unit, sign(Delta1+Delta2)
    -- both channels are updated together or not at all.
  - Hyperbolic RBM: an independent decision per channel, sign(Delta1) and
    sign(Delta2) -- each channel's weights can be updated selectively.

Energy convention (matches notebook 21): E(v,h) = -v^T W h - b_h^T h, no
visible bias. Since an RBM has no hidden-hidden connections, this energy
decomposes exactly hidden-unit by hidden-unit:

    E(v,h) = sum_j E_j(v,h),  E_j(v,h) = -h_j * (v @ W[:, j] + b_h[j])

Also reports user 3268's held-out test RMSE, real (channel-1-only
prediction) vs hyperbolic (channel-1 + channel-2 joint prediction), using
the same formulas as notebook 23.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
root = Path(__file__).resolve().parent
if root.name == "scripts":
    root = root.parent
proc = root / "data" / "processed"
out_dir = root / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

K = 10
RATING_LEVELS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
DEMO_USER_ID = 3268
SEED = 0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Load already-trained weights and data -- inference only, no retraining
# ---------------------------------------------------------------------------
W1 = np.load(proc / "rbmB1_weights_5k.npy")
bh1 = np.load(proc / "rbmB1_bias_hidden_5k.npy")
W2 = np.load(proc / "rbmB2_weights_5k.npy")
bh2 = np.load(proc / "rbmB2_bias_hidden_5k.npy")
channelB1 = np.load(proc / "channelB1_softmax.npy", mmap_mode="r")
channelB2 = np.load(proc / "channelB2_personality.npy", mmap_mode="r")
mask = np.load(proc / "mask.npy", mmap_mode="r")
cohort_user_ids = np.load(proc / "cohort_user_ids.npy").astype(int)
movie_vocab = np.load(proc / "movie_vocab.npy").astype(int)
user_means = np.load(proc / "user_means.npy").astype(np.float64)
test_labels = pd.read_csv(proc / "test_labels.csv")
mu_global = float(np.load(proc / "mu_global.npy"))

n_users, n_movies, _ = channelB1.shape
n_hidden = bh1.shape[0]
assert W1.shape == (n_movies * K, n_hidden)
assert W2.shape == (n_movies, n_hidden)

user_to_row = {int(uid): i for i, uid in enumerate(cohort_user_ids)}
movie_to_col = {int(mid): j for j, mid in enumerate(movie_vocab)}
W1_r = W1.reshape(n_movies, K, n_hidden)

demo_row = user_to_row[DEMO_USER_ID]
mask_col = np.asarray(mask[demo_row]).astype(np.float64)  # (n_movies,), 1 at rated movies

v1_0 = np.asarray(channelB1[demo_row]).reshape(-1).astype(np.float64)  # (n_movies*K,)
v2_0 = np.asarray(channelB2[demo_row]).reshape(-1).astype(np.float64)  # (n_movies,)

print(f"User {DEMO_USER_ID}: row={demo_row}  rated movies={int(mask_col.sum())}  "
      f"user_mean={user_means[demo_row]:.4f}  personality_true={user_means[demo_row] - mu_global:+.4f}")

# ---------------------------------------------------------------------------
# Step 1 -- positive phase (real data v0 -> h0), both channels
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

I1_0 = v1_0 @ W1 + bh1  # (n_hidden,), includes hidden bias -- this IS the per-unit energy signal
h1_prob_0 = sigmoid(I1_0)
h1_0 = (rng.random(h1_prob_0.shape) < h1_prob_0).astype(np.float64)

I2_0 = v2_0 @ W2 + bh2
h2_prob_0 = sigmoid(I2_0)
h2_0 = (rng.random(h2_prob_0.shape) < h2_prob_0).astype(np.float64)

# ---------------------------------------------------------------------------
# Step 2 -- reconstruction (h0 -> v1 -> h1), both channels
# ---------------------------------------------------------------------------
# channel 1: softmax reconstruction per movie over its K rating bins (Salakhutdinov, Mnih & Hinton 2007),
# masked to rated movies only, same convention as notebook 21
J_v1 = (h1_0 @ W1.T).reshape(n_movies, K)
v1_1_full = softmax(J_v1, axis=1)
v1_1 = (v1_1_full * mask_col[:, None]).reshape(-1)

I1_1 = v1_1 @ W1 + bh1
h1_prob_1 = sigmoid(I1_1)

# channel 2: scalar (Bernoulli-style) reconstruction per movie, masked to rated movies only
J_v2 = h2_0 @ W2.T
v2_1 = sigmoid(J_v2) * mask_col

I2_1 = v2_1 @ W2 + bh2
h2_prob_1 = sigmoid(I2_1)

# ---------------------------------------------------------------------------
# Per-hidden-unit energy change from one CD-1 step, split by channel
#   E_j(v, h) = -h_j * I_j(v)   =>   Delta_c[j] = E_j(v1, h1_prob) - E_j(v0, h0)
#                                              = h0[j] * I_0[j] - h1_prob[j] * I_1[j]
# Positive Delta = reconstruction energy higher than data energy = this
# channel's hidden unit favors the real data (a "good" update direction).
# ---------------------------------------------------------------------------
Delta1 = h1_0 * I1_0 - h1_prob_1 * I1_1
Delta2 = h2_0 * I2_0 - h2_prob_1 * I2_1

# ---------------------------------------------------------------------------
# Incomparable ratio: hidden units where the two channels disagree in sign
# ---------------------------------------------------------------------------
sign1 = np.sign(Delta1)
sign2 = np.sign(Delta2)
incomparable = sign1 != sign2
incomparable_ratio = incomparable.mean()

# ---------------------------------------------------------------------------
# Two acceptance strategies
#   Real RBM:        one scalar decision per unit, sign(Delta1 + Delta2)
#   Hyperbolic RBM:   independent per-channel decisions, sign(Delta1), sign(Delta2)
# ---------------------------------------------------------------------------
combined = Delta1 + Delta2
real_accept = combined > 0  # whole-unit accept/reject, both channels move together

hyp_accept1 = Delta1 > 0
hyp_accept2 = Delta2 > 0

# "Rescued": among incomparable units, a channel whose own signal is good
# (Delta_c > 0) but that the real RBM's combined rule would have discarded
# anyway (real_accept == False, because the other channel's larger negative
# Delta dragged the sum below zero). The selective strategy keeps this
# channel's update; the real RBM would have thrown it away along with the bad
# channel.
rescued_ch1 = incomparable & (Delta1 > 0) & (~real_accept)
rescued_ch2 = incomparable & (Delta2 > 0) & (~real_accept)
rescued_mask = rescued_ch1 | rescued_ch2
n_incomparable = int(incomparable.sum())
rescued_rate = (rescued_mask.sum() / n_incomparable) if n_incomparable > 0 else float("nan")

# Complementary case, reported for context: among incomparable units where the
# real RBM's combined rule accepts BOTH channels (real_accept == True), the
# channel with the negative Delta is a "bad" update the real RBM wrongly lets
# through; the selective strategy avoids updating it.
avoided_harm_ch1 = incomparable & (Delta1 < 0) & real_accept
avoided_harm_ch2 = incomparable & (Delta2 < 0) & real_accept
avoided_harm_mask = avoided_harm_ch1 | avoided_harm_ch2
avoided_harm_rate = (avoided_harm_mask.sum() / n_incomparable) if n_incomparable > 0 else float("nan")

print()
print(f"incomparable ratio (Delta1, Delta2 opposite sign): {incomparable_ratio:.4f}  "
      f"({n_incomparable} / {n_hidden} hidden units)")
print(f"rescued update rate (good channel saved from the real RBM's wrongful "
      f"combined rejection): {rescued_rate:.4f}")
print(f"avoided-harm rate (bad channel kept out despite the real RBM's wrongful "
      f"combined acceptance): {avoided_harm_rate:.4f}")

print()
print("acceptance strategy comparison, all hidden units:")
print(f"  real RBM      : accept-both={int(real_accept.sum())}  reject-both={int((~real_accept).sum())}")
print(f"  hyperbolic RBM: accept ch1={int(hyp_accept1.sum())}  accept ch2={int(hyp_accept2.sum())}  "
      f"accept-both={int((hyp_accept1 & hyp_accept2).sum())}  reject-both={int((~hyp_accept1 & ~hyp_accept2).sum())}")

# ---------------------------------------------------------------------------
# Held-out test RMSE for user 3268, real (channel 1 only) vs hyperbolic
# (channel 1 + channel 2), using the trained model's Step-1 hidden
# representation (h1_prob_0, h2_prob_0) -- same prediction formulas as
# notebook 23.
# ---------------------------------------------------------------------------
demo_test = test_labels[test_labels["userId"] == DEMO_USER_ID].copy()
demo_test["movie_col"] = demo_test["movieId"].map(movie_to_col)
demo_test = demo_test.dropna(subset=["movie_col"]).copy()
demo_test["movie_col"] = demo_test["movie_col"].astype(int)
movie_cols = demo_test["movie_col"].values

J1_test = W1_r[movie_cols] @ h1_prob_0  # (n_test, K)
sig1_test = sigmoid(J1_test)
denom_test = sig1_test.sum(axis=1)
denom_test = np.where(denom_test > 0, denom_test, 1.0)
r1_pred = (sig1_test * RATING_LEVELS).sum(axis=1) / denom_test

J2_test = W2[movie_cols] @ h2_prob_0  # (n_test,)
v_recon2_test = sigmoid(J2_test)
r_joint_pred = np.clip(r1_pred + v_recon2_test, 0.5, 5.0)

true_rating = demo_test["rating"].values
rmse_real = float(np.sqrt(np.mean((r1_pred - true_rating) ** 2)))
rmse_hyp = float(np.sqrt(np.mean((r_joint_pred - true_rating) ** 2)))

print()
print(f"User {DEMO_USER_ID} held-out test RMSE ({len(demo_test)} test movies):")
print(f"  {'strategy':<12s}{'RMSE':>10s}")
print(f"  {'real':<12s}{rmse_real:>10.4f}")
print(f"  {'hyperbolic':<12s}{rmse_hyp:>10.4f}")
print(f"  {'delta':<12s}{rmse_hyp - rmse_real:>+10.4f}")

# ---------------------------------------------------------------------------
# Plot: Delta1 vs Delta2, one point per hidden unit, quadrant-colored,
# rescued units highlighted
# ---------------------------------------------------------------------------
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]

QUADRANT_COLORS = {
    "Q1 both good (both accept)": "#5b7c99",
    "Q2 both bad (both reject)": "#b4654a",
    "Q3 e- bad only (incomparable)": "#c9a13b",
    "Q4 e+ bad only (incomparable)": "#7a9a5b",
}


def classify(d1, d2):
    q = np.full(len(d1), "", dtype="<U40")
    q[(d1 > 0) & (d2 > 0)] = "Q1 both good (both accept)"
    q[(d1 < 0) & (d2 < 0)] = "Q2 both bad (both reject)"
    q[(d1 > 0) & (d2 < 0)] = "Q3 e- bad only (incomparable)"
    q[(d1 < 0) & (d2 > 0)] = "Q4 e+ bad only (incomparable)"
    return q


quadrant = classify(Delta1, Delta2)

fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for q, color in QUADRANT_COLORS.items():
    sub = quadrant == q
    ax.scatter(Delta1[sub], Delta2[sub], color=color, s=40, alpha=0.85, zorder=3,
               edgecolors=SURFACE, linewidths=0.5, label=f"{q} (n={int(sub.sum())})")

# highlight rescued units (selective strategy saves a good channel the real RBM would drop)
ax.scatter(Delta1[rescued_mask], Delta2[rescued_mask], facecolors="none", edgecolors=INK_PRIMARY,
           s=110, linewidths=1.3, zorder=4, label=f"rescued by selective update (n={int(rescued_mask.sum())})")

lim = max(np.abs(Delta1).max(), np.abs(Delta2).max()) * 1.1
ax.plot([-lim, lim], [lim, -lim], color=INK_PRIMARY, linewidth=1.4, linestyle="--", zorder=2,
        label=r"real RBM boundary: $\Delta_1+\Delta_2=0$")
ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
ax.text(lim * 0.97, -lim * 0.95, r"hyperbolic RBM boundaries: $\Delta_1=0,\ \Delta_2=0$",
        ha="right", fontsize=8.5, color=INK_SECONDARY)

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel(r"$\Delta_1$ (e+ channel, per hidden unit)", color=INK_SECONDARY, fontsize=10)
ax.set_ylabel(r"$\Delta_2$ (e- channel, per hidden unit)", color=INK_SECONDARY, fontsize=10)
ax.set_title(f"User {DEMO_USER_ID}: one CD-1 step, {n_hidden} hidden units -- "
             f"real vs. selective (hyperbolic) update", color=INK_PRIMARY, fontsize=11, loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(BASELINE)
ax.tick_params(colors=INK_MUTED, labelsize=9)
ax.grid(color=GRID, linewidth=0.6, zorder=0)
plt.tight_layout()

out_path = out_dir / f"user{DEMO_USER_ID}_selective_update.png"
plt.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print(f"\nsaved chart to {out_path}")
