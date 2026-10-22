# CLAUDE.md

## Project

Comparative study of Restricted Boltzmann Machines across four number systems: ℝ (real), ℂ (complex), 𝔻 (hyperbolic), Fz (fuzzy), on MovieLens 20M. The primary deliverable is the live page rbm_number_systems.html. The page is the source of truth for progress.

## Roles

Yixuan sets every standard for this work. This session handles both requirements discussion and execution. The GitHub repo is the single source of truth. The local VS Code session is the only writer; every task ends with commit and push to main.

## Research standards

* Theory must be faithfully realized in code.
* The e₋ channel carries no preset meaning (no personality, no deviation, no semantic label). Its meaning is only what the model learns, interpreted afterwards.
* Math explanations never skip steps: write out every intermediate matrix operation, map every symbol to its concrete value before use, declare every variable, no obscure metaphors.
* Locked terminology: ℝ is ℝ, no “classical” modifier. Hyperbolic unit is k, with z = a + bk and k² = +1. Use “rated movie list”, never “mask” or “masking”. “Loss” means the gap between predicted and actual ratings.

## Page standards

* Portfolio grade, suitable for LinkedIn and YouTube; no personal names anywhere.
* The live page link is fixed: https://yixuan116.github.io/MovieLens-with-Restricted-Boltzmann-machine/rbm_number_systems.html. Never change it or add anything to it; to see a new deploy, use Cmd + Shift + R.
* Every figure already produced is used on the page, embedded inline in its section’s narrative, explained as it appears. No figure dumps at the end, no separate validation section repeating earlier content.
* Only final predictions and hidden feature visualizations go on the page; intermediate data stays off.
* Scannable, no long paragraphs.
* Interpretation goes in a short Meaning column or one short line, in plain terms; never restate numbers already shown, never add a summary paragraph under a table.
* Every symbol in a formula is declared on the page. Keep the formulas clean and put the declarations in a collapsible “Symbols” toggle right under the formula.
* When a step reuses something defined earlier, state what it is right there (e.g. “k, the exchange matrix K = [[0, 1], [1, 0]]”). Never write “see Step X” or “from section Y” in place of the definition.
* Page previews are reviewed in a separate chat before the prompt is sent; apply page edits directly, then commit and push. One change at a time. Deletions come last.
* In 01e and later sections, + channel values use #6f9bc2 and − channel values use #d98a3d.

## Execution rules

* Never hard code notebook numbers or file names: scan notebooks and outputs, create the next notebook after the highest existing number.
* Every task ends with commit and push to main.
* No screenshots or temporary verification files in the repo.
* At the end of each working day, update the Progress section below, then commit and push.
* After that end of day push, remind Yixuan to run git pull on the local copy.

## Communication

* Conversation is in Chinese; code, notebooks and page content are in English.
* In chat, explain in the plainest words possible; everyday analogies (like calling (v, h) a snapshot) are welcome; introduce no new jargon. The page itself uses the field's standard terms.
* In chat, write every vector or matrix out in full as a rendered LaTeX matrix (bmatrix in $$ … $$), entries shown, row vectors as one row and column vectors as one column. If a size is needed, put it directly under that matrix; never in a separate size line.
* Conclusions first. Concise; never restate what is already settled. One recommendation, one reason; no option menus.
* No dashes of any kind. No “not X but Y” constructions; state Y directly. Never open with “however”.
* Align scope before producing anything. One step at a time; never bundle steps that depend on results not yet seen.
* Never list future to-dos or open items unprompted; Yixuan decides what comes next.

## Data

* Dense subset of 1000 users × 1000 movies, density 61.55%. User funnel: rating count in [100, 3000], genre coverage ≥ 8 of 19, timestamp span > 1 year, top 1000 by activity. Movies: top 1000 by rated count within the subset.
* Split: chronological per user, earliest 80% train, latest 20% test, identical for all four number systems.
* Input mappings:
  * ℝ: a = (rating − 0.5) / 4.5.
  * 𝔻 (section 04): z = a + bk, b = popularity rank in the subset mapped to [−1, +1].
  * ℂ: z = a · e^(iθ), θ = 2π × weekday / 7.
  * Fz: Gaussian membership centered at the movie mean, five anchors at 1 to 5 stars.

## Progress (as of 2026-10-22)

* 01 Overview: four lenses and the H = 4 cross cutting test defined.
* 01b Same Rating, Four Lenses: input probe for all four systems on The Matrix vs Rumble in the Bronx, complete.
* 01c Hyperbolic Weights: Steps 1 to 8 built. Real input, hyperbolic weights W = w + v·k, diagonalized into W₊ = w + v and W₋ = w − v; σ applied per channel. Prediction = (v̂₊ + v̂₋)/2; b̂ = (v̂₊ − v̂₋)/2 kept as a structural signal. CD-1, lr 0.05, 50 epochs, batch 100 users, 128 hidden units per channel.
  * Step 2: diagonalization with P written out in three steps; collapsible "Why the two channels stay separate".
  * Step 5: hidden units drawn as on/off states s ∈ {0, 1} with P(s = 1) = h, matching CD-1 in code.
  * Step 6: general formula, Symbols toggle, diagonal form, + channel matrix form with sizes, entry i; reconstruction uses s.
  * Step 8: worked example table now shows formula and computation per row for W₊ and W₋ (one user, starting weights, c = d = 0); Symbols toggle.
  * Source of the Step 8 numbers: notebooks/31_step8_minus_channel.ipynb and outputs/31_step8_results.json (+ channel reuses notebook 26's ℝ baseline, seed 44; − channel seed 45).
* 01d Energy Change: Which Channel to Update: new section, Steps 1 to 5 built on the Rumble in the Bronx × h₁ example.
  * Step 1: energy of a configuration, partition function Z, ratio of the two CD-1 configurations where Z cancels.
  * Step 2: energy change Δ = w · Δw / lr; Δ₁ = +0.00079, Δ₂ = −0.00125; moving values colored.
  * Step 3: ΔE = Δ₁e₊ + Δ₂e₋ assembled with P in three steps; ΔE = −0.00023 + 0.00102k.
  * Step 4: kΔE = Δ₁e₊ − Δ₂e₋, four cell table, ℝ sees only Re(ΔE); the example falls in cell 3 (update e₋ only).
  * Step 5: gated update with switches g₊, g₋; all four cells for the example, cell 3 actual.
* 01e Fully Hyperbolic RBM: Steps 1 to 6 built on the same Rumble in the Bronx × h₁ example (user 24127, seeds 44/45).
  * b switched from movie popularity to user rating habit: b = (μ_u − 0.5)/4.5, μ_u = this user's mean rating over their training ratings only. b depends only on the user; added to every movie in the user's rated movie list in both channels (+b in channel +, −b in channel −), unrated movies stay 0. Current worked-example value: μ_u = 3.4557, b = 0.6568.
  * Step 1: v = a + bk; Step 2: W₊/W₋ and c₊/c₋ reused from 01c, link back not re-derived, x recomputed via b × Σ over the rated movie list R (|R| = 530); Step 3: q = q₊e₊ + q₋e₋; Step 4: four joint hidden-state probabilities; Step 5: v̂ reconstruction, now 4/128 (+) and 3/128 (−) hidden bits differ from 01c since b touches all 530 rated entries, not one; Step 6: CD-1 update ΔW = ΔW₊e₊ + ΔW₋e₋ = ΔA + ΔBk.
  * Source: notebooks/37_step1to6_rating_habit_b.ipynb and outputs/37_step1to6_rating_habit_b_results.json, reusing notebooks 31/32/33/34/35/36's exact seeds, weights and procedure; 01c column cross-checked against those notebooks' stored values.
* 02 Data Selection: live. Known issue: funnel top shows "0 users × 0 movies" and density shows 0.00%. Also verify the "138,493 qualified users" figure against the funnel logic.
* 03 Chronological split: live.
* 04 Hyperbolic Decoupling (input encoding version): live. ℝ RMSE 0.8431, MAE 0.6481; 𝔻 RMSE 0.9523, MAE 0.7187.

## Next

Notebook 32: run the four quadrant rule on full training data and plot the share of each cell over the 50 epochs, as the result figure closing 01d. Decide first with Yixuan: (1) per hidden unit or per single weight; (2) rebuild the data in this environment (MovieLens 20M, then notebook 25) with notebook 31's settings.
