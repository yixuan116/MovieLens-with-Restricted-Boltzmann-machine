# Emergent Factor Discovery — Real RBM Hidden Unit Interpretation (5k cohort)

## Context

The current "personality" channel (`channelB2_personality.npy` = μᵢ − μ_global, built in [12b_encodingB_personality_prep.ipynb](../../../Users/yixuan/Boltzmann%20Machine%20in%20Movie%20Lens/rbm-recsys/notebooks/12b_encodingB_personality_prep.ipynb)) is a hand-designed scalar, not something the model discovered. In [14b_evaluation_5k.ipynb](../../../Users/yixuan/Boltzmann%20Machine%20in%20Movie%20Lens/rbm-recsys/notebooks/14b_evaluation_5k.ipynb) the resulting "Hyperbolic joint" model beats the plain Real RBM overall (RMSE 0.9027 vs 0.9423) and for generous/middle users, but **regresses for strict users** (RMSE 1.3010 vs 1.2731) — evidence the hand-picked personality scalar doesn't capture what actually drives that group's ratings.

Goal of this work: stop assuming what "personality" is. Instead, treat the already-trained Real RBM (B1-only, never shown any personality signal) as ground truth for "what the model actually learned," and interpret its 128 hidden units post-hoc using real MovieLens metadata (movie side) and derived behavioral statistics (user side) — then use an ablation experiment to test whether the existing hyperbolic architecture is earning its keep beyond a naive per-user bias correction.

Theoretical grounding (confirmed via full read of the `Paper/` folder): Rumelhart, Hinton & Williams (1986, *Nature*) is the direct precedent for "hidden units self-organize into interpretable features, discoverable post-hoc." The specific movie/user metadata-correlation probing method has no exact precedent in the folder — closest are Koren/Bell/Volinsky (2009, recsys latent-factor interpretation) and Bau et al. (2017, Network Dissection), which should be cited as the methodological ancestors of this approach, though neither PDF is currently in `Paper/` (external references — see table below). The existing Kobayashi/Alpay hyperbolic-BM papers remain the correct citations for the *architecture* (why two channels can be trained independently and recombined), reframed per the user's own correction: they establish theoretical *legitimacy*, not empirically proven *maximization*.

### Theoretical references (which paper backs which claim)

| Claim in this plan | Paper (title, authors, year) | Source file |
|---|---|---|
| Hidden units self-organize into interpretable, task-relevant features, discoverable post-hoc by inspecting weights/activations — the core premise of Parts 2–5 | Rumelhart, Hinton & Williams, "Learning Representations by Back-Propagating Errors," *Nature* 323 (1986) | `Paper/Rumelhart_et_al-1986-Nature.pdf` |
| Boltzmann-family hidden units capture a domain's statistical regularities without supervision (one level down from Rumelhart — general BM learning theory) | Ackley, Hinton & Sejnowski, "A Learning Algorithm for Boltzmann Machines," *Cognitive Science* 9 (1985) | `Paper/C. Classic C hinton A Learning Algorithm.pdf` |
| Boltzmann-Gibbs distribution / simulated-annealing learning-rule formalism underlying any CD-trained RBM (incl. B1) | Hinton & Sejnowski, "Learning and Relearning in Boltzmann Machines," ch. 7 in *Parallel Distributed Processing* Vol. 1 (MIT Press, 1986) | `Paper/Classic Hinton_Learning and Relearning in Boltzmann Machines 1986-3239.pdf` |
| Energy-based associative-memory model that the Boltzmann Machine (a stochastic Hopfield net with hidden units) derives from | Hopfield, "Neural Networks and Physical Systems with Emergent Collective Computational Abilities," *PNAS* 79 (1982) | `Paper/Hopfield_pnas_82.pdf` |
| Softmax-visible-unit RBM-CF architecture that Channel B1 is built on (visible rating units, binary hidden units, CD training) | Salakhutdinov, Mnih & Hinton, "Restricted Boltzmann Machines for Collaborative Filtering," ICML (2007) | `Paper/netflix.pdf` |
| Hyperbolic-valued neuron state + energy function — theoretical precedent for combining real+personality channels as a hyperbolic pair (existing architecture, not this plan's new work) | Kobayashi, "Hyperbolic Hopfield Neural Networks," *IEEE TNNLS* 24(2) (2013) | `Paper/Hyperbolic_Hopfield_Neural_Networks.pdf` |
| Hyperbolic Hopfield nets with directional multistate activation — further hyperbolic-state precedent | Kobayashi, "Hyperbolic Hopfield Neural Networks with Directional Multistate Activation Function," *Neurocomputing* 275 (2018) | `Paper/A. 3_Hyperbolic-Hopfield-neural-networks-with-directional-multist_2018_Neurocompu.pdf` |
| Information geometry of hyperbolic-valued Boltzmann Machines (Fisher metric, dual parameters) — the direct predecessor the user's own paper extends/critiques | Kobayashi, "Information Geometry of Hyperbolic-Valued Boltzmann Machines," *Neurocomputing* 431 (2021) | `Paper/A. Hyperbolic A. Kobayashi_boltzmann nformation geometry of hyperbolic-valued Boltzmann machines.pdf` |
| Axiomatic hyperbolic-valued probability measure (Kolmogorov generalization) — foundation the user's own hyperbolic-RBM paper builds on | Alpay, Luna-Elizarrarás & Shapiro, "Kolmogorov's Axioms for Probabilities with Values in Hyperbolic Numbers," *Advances in Applied Clifford Algebras* (2016) | `Paper/241_ALS_Kolmogorov Axioms for Prob Hyperbolic.pdf` |
| Formal probabilistic backbone for the existing real+personality hyperbolic joint channel (hyperbolic Hammersley-Clifford theorem, componentwise CD-1) — **the user's own paper/draft** | Jing (Yixuan Jing), "A Hyperbolic-Valued Probability Framework for Restricted Boltzmann Machines via Idempotent Decomposition," Chapman University draft, Spring 2026 (advisor: Daniel Alpay) | `Paper/A H valued Prob for RBM Yixuan Jing.pdf` |
| Course context / reading list this whole line of work sits inside | "Restricted Boltzmann Machines: Classical and Hyperbolic Numbers" — course syllabus, Chapman University, Spring 2026 | `Paper/Research Syllabus.pdf` |
| Classic recsys practice of interpreting latent factors by inspecting top-loading items (e.g. "serious vs. escapist") — closest precedent for Part 2's movie-side interpretation | Koren, Bell & Volinsky, "Matrix Factorization Techniques for Recommender Systems," *IEEE Computer* 42(8) (2009) | **not in `Paper/`** — external reference, add PDF if this citation is used in the write-up |
| Quantitative unit-concept alignment/correlation methodology for interpreting hidden units — closest methodological ancestor for Parts 2 & 4 as a whole | Bau, Zhou, Khosla, Oliva & Torralba, "Network Dissection: Quantifying Interpretability of Deep Visual Representations," CVPR (2017) | **not in `Paper/`** — external reference, add PDF if this citation is used in the write-up |

Not used as theoretical support for this plan (checked, ruled out): `CAPB_2nd.pdf` (Alpay's complex-analysis textbook — no hyperbolic-number or ML content), `1-s2.0-S1051200421001536-main.pdf` (signal-processing "hyperbolic frequency modulation" — unrelated "hyperbolic," refers to chirp signal shape, not hyperbolic-number algebra), and the three Alfsmann DSP papers (`Alfsmann_*.pdf`) — pure hypercomplex-algebra/DSP background, cite only if the write-up needs the underlying idempotent-decomposition math itself, not for any claim in this plan.

## Confirmed scope (from discussion)

- **Cohort**: reuse the existing 5,000-user / 13,129-movie cohort from [09b_data_prep_5k.ipynb](../../../Users/yixuan/Boltzmann%20Machine%20in%20Movie%20Lens/rbm-recsys/notebooks/09b_data_prep_5k.ipynb) unchanged. Selection criterion was activity (≥20 ratings, iterative core filter, top 5k by count) — **not** rating extremity, so extreme raters are already naturally present in-sample. The 51-user/2,000-movie cohort in notebooks 15/16 is unrelated old work and is not touched.
- **Train/test split**: unchanged 80/20 temporal-per-user split. All discovery analysis runs on **train-partition data only** (`mask==1`, `channelB1_softmax.npy`) to stay consistent with how `H1`, `personality.npy`, and `user_means.npy` were already derived — this avoids leaking test-period ratings into the "what did the model learn" analysis.
- **Base model**: existing Real RBM only (`rbmB1_weights_5k.npy`, `rbmB1_bias_hidden_5k.npy`). No retraining.
- **Movie-side evidence**: genres + release year (`movie.csv`) + genome tag relevance (`genome_scores.csv`/`genome_tags.csv`) + user free-text tags (`tag.csv`). No director/actor/soundtrack fields exist in this dataset (would require external TMDB calls via `link.csv` — explicitly out of scope for now).
- **User-side candidate factors** (kept separate, not pre-merged, since mean and variance are statistically independent axes): leniency (μᵢ−μ_global), extremity (rating std), activity (rating count), genre-preference entropy, contrarian score (signed bias + unsigned degree, vs. each movie's population mean), popularity bias (avg population rating-count of rated movies).
- **Method**: Pearson correlation as the primary screening tool (cheap, matches project's existing analytical style, has real precedent) — with explicit acknowledgment of its blind spots (linear-only, no confound control, associational not causal). Partial correlation is used to control confounds; a probing regression checks whether a factor is concentrated in one unit or distributed across many.
- **Ablation**: test whether the hyperbolic joint's improvement comes from the algebraic structure itself or just from "having any second per-user bias signal," by comparing against a naive learned-bias baseline.

## Reusable code (exact sources — port, don't rewrite)

| Reused piece | Source | Adaptation needed |
|---|---|---|
| `movie_importance(W, k)`, `top_movies_for_unit(W, k, n)` | `14_hyperbolic_quadrant_encodingB_analysis_and_viz.ipynb` Part 7 | Repoint to `rbmB1_weights_5k.npy`, `n_movies=13129`; run over all 128 units instead of a top-5 subset |
| Genre-frequency counting (`genre_counts`) + tag aggregation via `Counter` over `tag.csv` | same, Part 7 | Port as-is; **add** a corpus-wide baseline frequency to compute enrichment ratio (nb14 compared two channels against each other — we only have one channel now, so compare each unit's top-pool genre mix against the full-vocab genre mix instead) |
| Per-user `contrarian_degree` / `contrarian_bias` / `rating_std` / `rating_count` / `mean_rating` (population mean streamed from full `rating.csv`, filtered to `movie_vocab`) | same, Part 8 | Port the population-mean streaming logic as-is; **replace** the "user's own ratings" source — instead of re-streaming `rating.csv` with timestamp cutoffs, decode train-only ratings directly from `channelB1_softmax.npy` + `mask` (already exactly the train split; avoids redundant/inconsistent re-derivation) |
| `partial_corr(x, y, control)` | same, Part 9 | Reuse verbatim for confound control (e.g., controlling extremity/contrarian correlations for activity level) |
| Pearson-correlation-with-strong/moderate/weak bucket + scatter+regression-line plotting pattern | same, Parts 8–9 | Adapt to loop over 128 `H1` columns instead of a single `divergence` scalar |
| `H1 = sigmoid(X1 @ W1 + bh1)` forward pass | `14b_evaluation_5k.ipynb` | Reuse as-is |
| `personality.npy`, `user_means.npy`, `mu_global.npy` | `data/processed/` | Reuse as leniency feature, no recomputation |
| Existing eval baseline (Real RBM / Hyperbolic joint RMSE·MAE, overall + group breakdown) | `14b_evaluation_5k.ipynb`, `outputs/evaluation_summary_5k.csv` | Reused as the "before" comparison row in the new ablation table |

## New notebook: `notebooks/17_emergent_factor_discovery_5k.ipynb`

**Part 0 — Setup**: load `W1`, `bh1`, `channelB1_softmax.npy`, `mask.npy`, `cohort_user_ids.npy`, `movie_vocab.npy`, `user_means.npy`, `personality.npy`, `mu_global.npy`; recompute `H1` (copied from 14b).

**Part 1 — Movie-side metadata assembly**
- Parse release year from `movie.csv` titles via regex (`\((\d{4})\)$`); build `movie_meta` (title, genres list, year) indexed by `movieId`, restricted to `movie_vocab`.
- Load `genome_scores.csv` + `genome_tags.csv`, filter to `movie_vocab`, pivot into a dense `(13129, ~1128)` relevance matrix aligned to `movie_vocab` order.
- Aggregate `tag.csv` free-text tags per movie (`Counter.most_common(5)`), filtered to `movie_vocab`.

**Part 2 — Movie-side hidden-unit interpretation (all 128 units)**
- Per unit: `top_movies_for_unit(W1, k, n=20)`.
- Genre evidence: enrichment ratio of top-pool genre mix vs. corpus-wide genre mix.
- Year evidence: mean/median year of top-pool vs. corpus-wide mean year (era skew).
- Genome-tag evidence: **vectorized** Pearson correlation between the unit's full `(13129,)` movie-importance vector and every tag's relevance column (avoid a 128×1128 Python double-loop; use the vectorized correlation formula) — keep top-10 tags per unit by |r|.
- User-tag evidence: most common free-text tags among the top-pool.
- Output one "interpretation card" per unit → `outputs/hidden_unit_movie_cards_5k.csv`.

**Part 3 — User-side behavioral feature table**
- leniency = `personality.npy` (reused, no recompute).
- extremity = per-user train-rating std (decoded from `channelB1_softmax.npy`/`mask`).
- activity = per-user train-rating count (`mask.sum(axis=1)`).
- genre-preference entropy = Shannon entropy of each user's train-rated-movie genre distribution (new, uses `movie_meta` from Part 1).
- contrarian score (signed `contrarian_bias`, unsigned `contrarian_degree`) = user's train rating − that movie's population mean (population mean streamed from full `rating.csv`, filtered to `movie_vocab`, matching nb14's precedent — population is the general MovieLens base, not just the 5k cohort, to avoid circularity).
- popularity bias = avg population rating-count of the user's train-rated movies.

**Part 4 — Correlation analysis (user-side)**
- 128 units × 6 factors: Pearson r + p, strong/moderate/weak bucket (reused pattern).
- Partial correlation (`partial_corr`, controlling for `activity`) for extremity/contrarian/genre-entropy, since these could be confounded by how many ratings a user has.
- Probing regression: each factor ~ full 128-dim `H1` (ridge regression, report R²) to flag whether a factor is concentrated in one unit or distributed across many — directly answers the earlier "is extremity just leniency again?" question empirically.
- Output → `outputs/hidden_unit_user_factor_correlations_5k.csv` + a 128×6 correlation-strength heatmap (via the dataviz skill).

**Part 5 — Synthesis**
- Combine Part 2 + Part 4 per unit into a human-readable summary (e.g. "Unit 42: top genome tags {…}, genre enrichment {…}, year skew …, correlates with contrarian_bias r=0.52 → reads as a contrarian-appeal factor").
- Save → `outputs/hidden_unit_summary_5k.md`.

**Part 6 — Ablation: hyperbolic joint vs. naive learned bias**
- Existing baseline (reused, not recomputed): Real RBM RMSE/MAE = 0.9423/0.7537; Hyperbolic joint = 0.9027/0.6970 (overall + generous/strict/middle breakdown from 14b).
- New comparator: fit a simple ridge regression predicting a per-user bias term from `μ_user` (train-only) and add it to `r1`; evaluate on the same `test_labels.csv`, same group breakdown.
- Produce a 3-row comparison table (Real RBM / Hyperbolic joint / Real RBM + naive bias) — isolates whether the hyperbolic algebra contributes beyond a plain bias correction.
- Save → `outputs/ablation_hyperbolic_vs_bias_5k.csv`.

**Part 7 — Documentation**
- Append a short "Emergent Factor Discovery — Methodology & Citations" section to [docs/hhNN_paper_formulas.md](../../../Users/yixuan/Boltzmann%20Machine%20in%20Movie%20Lens/rbm-recsys/docs/hhNN_paper_formulas.md): cite Rumelhart et al. 1986, Koren/Bell/Volinsky 2009, Bau et al. 2017 for the probing methodology; restate the Kobayashi/Alpay citations with the corrected framing ("guarantees legitimacy of independent two-channel training," not "maximizes performance").

## Verification

- Run the notebook top to bottom; all shape/consistency asserts must pass (128 units, 13,129 movies, 5,000 users, ~1,128 genome tags).
- Sanity-check a few interpretable units by hand (e.g., confirm a unit whose top genome tags skew "animation/children" also shows a younger-than-average year skew and matching genre enrichment) before trusting the full 128-row table.
- Regression check: confirm Part 6's "Real RBM" and "Hyperbolic joint" rows exactly reproduce 14b's existing numbers (0.9423/0.7537 and 0.9027/0.6970) — if they don't match, something in data loading was altered and must be fixed before trusting the new "naive bias" comparator.
