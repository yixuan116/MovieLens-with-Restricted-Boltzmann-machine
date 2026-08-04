# Latent Feature Discovery via Hyperbolic-Valued RBM

Draft in progress. Structure follows the outline agreed 2026-08-03. Sections below are filled in as they're finalized; empty sections are placeholders.

## 1. Introduction

*(placeholder)*

## 2. Background

### 2.1 RBM as feature detector

*(placeholder — Rumelhart, Hinton & Williams 1986; Ackley, Hinton & Sejnowski 1985; Hinton & Sejnowski 1986. See `docs/17_emergent_factor_discovery_plan.md` theoretical-references table for the full citation list already compiled for the companion empirical study.)*

### 2.2 𝔻 framework recap

*(placeholder — Kobayashi 2013/2018/2021; Alpay, Luna-Elizarrarás & Shapiro 2016.)*

### 2.3 Entropy on 𝔻: formal definition of $H_{\mathbb{D}}$

**Step 1 — Algebraic structure of $\mathbb{D}$** (Alpay–Luna-Elizarrarás–Shapiro 2016, §2)

$$z = a + bk, \qquad k^2 = 1$$

$$e_+ = \tfrac{1}{2}(1+k), \qquad e_- = \tfrac{1}{2}(1-k)$$

$$e_+^2 = e_+, \qquad e_-^2 = e_-, \qquad e_+ e_- = 0$$

$$z = (a+b)e_+ + (a-b)e_- = \nu_1 e_+ + \nu_2 e_-$$

All algebraic operations are component-wise under this idempotent decomposition.

**Step 2 — Kolmogorov axioms on $\mathbb{D}$** (Alpay–Luna-Elizarrarás–Shapiro 2016, §3–5)

$P_{\mathbb{D}}: \Sigma \to \mathbb{D}^+$ satisfies $P_{\mathbb{D}}(\Omega) = 1$ (or $= e_+$ or $= e_-$), with countable additivity holding under the $\preceq_{\mathbb{D}}$ partial order.

Under the idempotent decomposition, $P_{\mathbb{D}}(A) = P_1(A)e_+ + P_2(A)e_-$, where $P_1, P_2$ are each classical real-valued probability measures (or inactive $=0$).

**Step 3 — $\log$ on $\mathbb{D}$** (Paper 1 §4.3.2; direct algebraic extension)

For $p = \alpha e_+ + \beta e_- \in \mathbb{D}^+$ interior ($\alpha, \beta > 0$):

$$\log p = (\log \alpha)e_+ + (\log \beta)e_-$$

Well-defined for $\alpha, \beta > 0$ — i.e., $p$ strictly interior, away from the zero-divisor locus.

**Step 4 — Definition and properties of $H_{\mathbb{D}}$** (Bory-Reyes, Molina-Fernández & Sigarreta-Almira 2026, Definition 5)

$$H_{\mathbb{D}}(P) = \Big(-\sum_j p_j^{(1)} \log p_j^{(1)}\Big)e_+ + \Big(-\sum_j p_j^{(2)} \log p_j^{(2)}\Big)e_- = H^{(1)}e_+ + H^{(2)}e_-$$

Key properties:

- **Schur-concavity** under $\preceq_{\mathbb{D}}$ (Proposition 3.2): $P \succeq_{\{\mathbb{D},\text{maj}\}} Q \implies H_{\mathbb{D}}(P) \preceq_{\mathbb{D}} H_{\mathbb{D}}(Q)$
- **Upper bound** $s^{(1)}\log(n)\cdot e_+ + s^{(2)}\log(n)\cdot e_-$, attained only at the uniform distribution (Corollary 3.3)
- **Interpretation in the $\{1,k\}$ basis**: real part $=\tfrac{1}{2}(H^{(1)}+H^{(2)})$, the average of the two channel entropies; $k$-part $=\tfrac{1}{2}(H^{(1)}-H^{(2)})$, half the difference between them

**Why this matters for the empirical study (§4.4):** the $k$-component of $H_{\mathbb{D}}$ — the entropy gap between $e_+$ and $e_-$ — is a direct, correlation-free test of whether the hyperbolic decomposition captures structure beyond a single real-valued RBM. A gap that is clearly nonzero is quantitative evidence the two channels see genuinely different structure; a gap near zero would mean $e_+$ and $e_-$ are redundant. This connects directly to the companion notebook study's open question (`docs/17_emergent_factor_discovery_plan.md`, Part 5b): PC1 (68% of the real-channel hidden variance) remains only ~42% explained by any combination of hand-defined behavioral/movie evidence tried so far. The entropy gap offers a way to ask whether that unexplained portion reflects genuine additional structure the $\mathbb{D}$-decomposition reveals, independent of correlation-based probing.

## 3. Method

### 3.1 Model

*(placeholder — $e_+$ channel encodes raw rating; $e_-$ channel encodes deviation from user mean (per-rating, not the broadcast-scalar construction used for the earlier leniency/extremity/PC1 channels in the companion notebook study); 5,000-user cohort; trained weights.)*

### 3.2 Feature extraction

*(placeholder — $W$ matrix, $H$ matrix, split by $e_+/e_-$.)*

### 3.3 Movie-side analysis

*(placeholder — cluster $W$ columns, annotate with genres and genome tags.)*

### 3.4 User-side analysis

*(placeholder — cluster $H$ rows using rating statistics, tag behavior, and per-channel entropy as clustering features.)*

### 3.5 Cross-channel comparison

Cluster $e_+$'s and $e_-$'s hidden units **separately**, producing two independent cluster structures. Compare the two channels at the level of cluster vocabulary — what categories of features each channel collectively discovers — rather than unit-to-unit. ($e_+$'s clusters may be entirely content-based — genre, era; $e_-$'s may show behavior-based structure.) No unit-index correspondence between channels is assumed or required, since $e_+$ and $e_-$ are trained independently.

## 4. Results

### 4.1 Movie-side: what each channel's hidden units detect

*(placeholder)*

### 4.2 User-side: user group structure (entropy as a discovery feature)

*(placeholder — depends on §2.3.)*

### 4.3 $e_+$ vs. $e_-$ divergence: qualitative structural differences

*(placeholder — follows §3.5's cluster-vocabulary comparison, not unit-level.)*

### 4.4 Entropy gap: $e_+$ vs. $e_-$ entropy difference as a measure of structure $\mathbb{D}$ reveals beyond $\mathbb{R}$

*(placeholder — depends on §2.3; see the note under §2.3 above for how this connects to the companion study's unresolved PC1 question.)*

### 4.5 $\mathbb{R}$ RBM baseline comparison

*(placeholder — reuse the companion notebook study's 5-way test-set comparison, `notebooks/18_pc1_deep_dive_and_validation_5k.ipynb`, Part 4/5.)*

## 5. Discussion and Future Work

*(placeholder)*
