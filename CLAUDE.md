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
* Every figure already produced is used on the page, embedded inline in its section’s narrative, explained as it appears. No figure dumps at the end, no separate validation section repeating earlier content.
* Only final predictions and hidden feature visualizations go on the page; intermediate data stays off.
* Scannable, no long paragraphs.
* Every symbol in a formula is declared on the page. Keep the formulas clean and put the declarations in a collapsible “Symbols” toggle right under the formula.
* Before any page edit, show a rendered preview of formulas and layout; edit only after Yixuan confirms. One change at a time. Deletions come last, after visual confirmation.

## Execution rules

* Never hard code notebook numbers or file names: scan notebooks and outputs, create the next notebook after the highest existing number.
* Every task ends with commit and push to main.
* No screenshots or temporary verification files in the repo.
* At the end of each working day, update the Progress section below, then commit and push.
* After that end of day push, remind Yixuan to run git pull on the local copy.

## Communication

* Conversation is in Chinese; code, notebooks and page content are in English.
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
* 01d Hyperbolic Weights: Steps 1 to 8 built. Real input, hyperbolic weights W = w + v·k, diagonalized into W₊ = w + v and W₋ = w − v; σ applied per channel. Prediction = (v̂₊ + v̂₋)/2; b̂ = (v̂₊ − v̂₋)/2 kept as a structural signal. CD-1, lr 0.05, 50 epochs, batch 100 users, 128 hidden units per channel. First update hand checked against the formula.
* 02 Data Selection: live. Known issue: funnel top shows “0 users × 0 movies” and density shows 0.00%. Also verify the “138,493 qualified users” figure against the funnel logic.
* 03 Chronological split: live.
* 04 Hyperbolic Decoupling (input encoding version): live. ℝ RMSE 0.8431, MAE 0.6481; 𝔻 RMSE 0.9523, MAE 0.7187.

## Next

Four quadrant table on top of 01d. PLACEHOLDER: scope to be discussed with Yixuan before any code. Concept: per unit, ΔE = Δ₁e₊ + Δ₂e₋, with kΔE distinguishing which channel is positive; (sign Δ₁, sign Δ₂) falls in one of four mutually exclusive cells: No flip, Flip Δ₂, Flip Δ₁, Flip both. Mixed sign cells staying above zero through training would show the 𝔻 structure is genuinely used.

Open items, order to be decided: ℂ and Fz full training pipelines; H = 4 squeeze test; fix section 02 display.
