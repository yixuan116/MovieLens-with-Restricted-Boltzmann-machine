# CLAUDE.md

## Project

Comparative study of Restricted Boltzmann Machines across four number systems: ℝ (real), ℂ (complex), 𝔻 (hyperbolic), Fz (fuzzy), on MovieLens 20M. The primary deliverable is the live page rbm_number_systems.html. The page is the source of truth for progress.

## Roles

Yixuan sets every standard for this work. This session handles both requirements discussion and execution. The GitHub repo is the single source of truth: this session is the only writer; local copies only pull.

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
* Every symbol in a formula is declared on the page. Keep the formulas clean and put the declarations in a collapsible “Symbols” toggle right under the formula.
* When a step reuses something defined earlier, state what it is right there (e.g. “k, the exchange matrix K = [[0, 1], [1, 0]]”). Never write “see Step X” or “from section Y” in place of the definition.
* Before any page edit, show a rendered preview of formulas and layout; edit only after Yixuan confirms. One change at a time. Deletions come last, after visual confirmation.

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

## Data

* Dense subset of 1000 users × 1000 movies, density 61.55%. User funnel: rating count in [100, 3000], genre coverage ≥ 8 of 19, timestamp span > 1 year, top 1000 by activity. Movies: top 1000 by rated count within the subset.
* Split: chronological per user, earliest 80% train, latest 20% test, identical for all four number systems.
* Input mappings:
  * ℝ: a = (rating − 0.5) / 4.5.
  * 𝔻 (section 04): z = a + bk, b = popularity rank in the subset mapped to [−1, +1].
  * ℂ: z = a · e^(iθ), θ = 2π × weekday / 7.
  * Fz: Gaussian membership centered at the movie mean, five anchors at 1 to 5 stars.

## Progress (as of 2026-10-13)

* 01 Overview: four lenses and the H = 4 cross cutting test defined.
* 01b Same Rating, Four Lenses: input probe for all four systems on The Matrix vs Rumble in the Bronx, complete.
* 01d Hyperbolic Weights: Steps 1 to 8 built. Real input, hyperbolic weights W = w + v·k, diagonalized into W₊ = w + v and W₋ = w − v; σ applied per channel. Prediction = (v̂₊ + v̂₋)/2; b̂ = (v̂₊ − v̂₋)/2 kept as a structural signal. CD-1, lr 0.05, 50 epochs, batch 100 users, 128 hidden units per channel.
  * Step 2: diagonalization with P written out in three steps; collapsible "Why the two channels stay separate".
  * Step 5: hidden units drawn as on/off states s ∈ {0, 1} with P(s = 1) = h, matching CD-1 in code.
  * Step 6: general formula, Symbols toggle, diagonal form, + channel matrix form with sizes, entry i; reconstruction uses s.
  * Step 8: worked example table now shows formula and computation per row for W₊ and W₋ (one user, starting weights, c = d = 0); Symbols toggle.
  * Source of the Step 8 numbers: notebooks/31_step8_minus_channel.ipynb and outputs/31_step8_results.json (+ channel reuses notebook 26's ℝ baseline, seed 44; − channel seed 45).
* 01e Energy Change: Which Channel to Update: new section, Steps 1 to 5 built on the Rumble in the Bronx × h₁ example.
  * Step 1: energy of a configuration, partition function Z, ratio of the two CD-1 configurations where Z cancels.
  * Step 2: energy change Δ = w · Δw / lr; Δ₁ = +0.00079, Δ₂ = −0.00125; moving values colored.
  * Step 3: ΔE = Δ₁e₊ + Δ₂e₋ assembled with P in three steps; ΔE = −0.00023 + 0.00102k.
  * Step 4: kΔE = Δ₁e₊ − Δ₂e₋, four cell table, ℝ sees only Re(ΔE); the example falls in cell 3 (update e₋ only).
  * Step 5: gated update with switches g₊, g₋; all four cells for the example, cell 3 actual.
* 02 Data Selection: live. Known issue: funnel top shows "0 users × 0 movies" and density shows 0.00%. Also verify the "138,493 qualified users" figure against the funnel logic.
* 03 Chronological split: live.
* 04 Hyperbolic Decoupling (input encoding version): live. ℝ RMSE 0.8431, MAE 0.6481; 𝔻 RMSE 0.9523, MAE 0.7187.

## Next

Notebook 32: run the four quadrant rule on full training data and plot the share of each cell over the 50 epochs, as the result figure closing 01e. Decide first with Yixuan: (1) per hidden unit or per single weight; (2) rebuild the data in this environment (MovieLens 20M, then notebook 25) with notebook 31's settings.

Open items, order to be decided: 01d Steps 1 to 4 still lack Symbols toggles; ℂ and Fz full training pipelines; H = 4 squeeze test; fix section 02 display; README and repo About, written last.
