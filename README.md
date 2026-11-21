# GTA 6 Trophy Rarity Forecast — all 55 PS5 trophies, week one

[![GTA 6 trophy rarity forecast](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rahullpc/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rahullpc/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A forecast of every PlayStation 5 trophy's earn rate on Exophase for Grand Theft Auto VI at 2026-11-26T00:00:00Z, built for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01). The program reads each case's history of Exophase reads and writes five quantiles (levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95) per trophy. Scoring is the mean pinball loss on the logit scale.

## The model

An earn rate is a ratio — earned players over tracked players — and both sides move. On the logit scale a trophy's rate drifts as the player base matures, so for each trophy the program fits a weighted straight line to logit(rate) against log(1 + hours since the case's first read) — log time because growth saturates — weighting recent reads more (24 h half-life), then anchors the forecast at the last observed rate and extrapolates the fitted slope, damped (0.35×), to the target. The spread combines a random-walk term (0.5 per root day, shrunk to 0.6×), the fit's residual scatter, and the uncertainty of carrying a slope over the horizon.

A second branch handles a frozen page: when every read in the history is identical (rates and players tracked unchanged for hours), Exophase has not refreshed and a refresh is overdue, so the forecast shifts down — a refresh dilutes every rate because the players it adds have earned almost nothing yet — by about 1.6 logit (less for common trophies, none for a trophy already at 100%), and widens the spread. A case the model cannot run falls back to the flat persistence forecast (last rate, random-walk spread) rather than crashing, so no case is ever invalid on our account.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, standard library only, no network). It reads every `*.in` in the input directory (`$MAKEFUN_INPUT_DIR`, else `/input`) and writes one `<id>.out` per case to the output directory (`$MAKEFUN_OUTPUT_DIR`, else `/output`). Output is at most 65,536 bytes per case, quantiles non-decreasing in 0–100, deterministic (two runs are byte-identical), and well under the 8-second limit.

## Files

- `main.py` — the entry program (Python 3.13, standard library only).
- `README.md` — this file.
- `LICENSE` — MIT.

## Validation (local)

- Practice cases p00/p01: valid (55 rows, monotone, 0–100), deterministic, ~0.09 s for both. The real practice answers are not in the kit, so the loss was measured against a refresh simulator calibrated to reproduce the house flat baseline's published practice loss (~1.51, typical miss ~0.68 points): this model scores ~0.15–0.20 there, about ten times better than the flat baseline.
- Synthetic final (a plausible f00-like history with saturating growth and periodic refreshes, 120 reads, horizon 6,510 min): loss 0.25 against the flat baseline's 0.34 and the curve baseline's 0.39, with a median logit error of 0.10. The advantage holds across 8 random seeds.
- Edge cases (single read, null rates, a 0% trophy, an empty history, a 30-day horizon, an empty input directory) all produce a valid file and never crash.
