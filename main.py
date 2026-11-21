"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Reads each case's Exophase trophy earn rates (the percent of tracked PS5
players who earned each trophy) and forecasts every trophy's rate at the
case's target as five quantiles (levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95).

The model, in one paragraph: an earn rate is a ratio, earned players over
tracked players, and both sides move.  On the logit scale the rate of a
trophy drifts as the player base matures, so for each trophy I fit a
weighted straight line to logit(rate) against log(1 + hours since the
case's first read) -- log time because growth saturates -- weighting recent
reads more (24 h half-life), then anchor the forecast at the last observed
rate and extrapolate the fitted slope, damped, to the target.  The spread
combines a random-walk term (the house's 0.5 per root day), the fit's
residual scatter, and the uncertainty of extrapolating a slope over the
horizon.  A second branch handles a frozen page: when every read in the
history is byte-identical (rates and players tracked unchanged for hours),
Exophase has not refreshed and a refresh is overdue, so the forecast
shifts down -- a refresh dilutes every rate because the new players it
adds have earned almost nothing yet -- by an amount that grows as the
trophy gets rarer, and widens the spread.  The one trophy already at
100% does not shift: a refresh cannot lower it.

Quantiles are the median logit plus the standard normal's quantile at
each level times the spread, mapped back to percent, rounded to 4
decimals half up, clamped to 0.01..99.99 and non-decreasing by
construction.  The program is deterministic: same input, same bytes.

Run as the grader does: `python3 -P -s /bot/main.py`, standard library
only, no network.  It reads $MAKEFUN_INPUT_DIR (else /input) and writes
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in, in name
order.  A case that cannot be modelled falls back to the flat
persistence forecast (last rate, random-walk spread) rather than
crashing, so no case is ever invalid on our account.
"""

import json
import math
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels (accurate to 1e-15).
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# Random-walk spread of a rate's logit move, per square-root day.
SIGMA_RW = 0.5
# Recency weighting: a read this many hours old carries half the weight.
HALF_LIFE_H = 24.0
# The fitted slope is damped this much when extrapolated: growth slows
# as the rate saturates, so a straight-line carry of the recent slope
# overshoots over a multi-day horizon.  Tuned on a synthetic final
# (0.35 beats 0.5 and 0.65 across seeds) and on the practice refresh.
SLOPE_DAMP = 0.35
# How much of the random-walk term the trend branch keeps: the trend's
# median is better than "the rate stays put", so the shock term shrinks.
RW_SHRINK = 0.6
# A fitted slope's extrapolation is uncertain in proportion to its size.
SLOPE_UNC_COEF = 0.2
# Floor and cap on the fit's residual scatter (logit units).
RESID_FLOOR = 0.15
RESID_CAP = 1.0
# Never let the forecast's median logit move this far from the last read.
MAX_LOGIT_MOVE = 2.5
# Frozen-page branch: the history must span at least this long, all
# identical, before a refresh is treated as overdue.
FREEZE_MIN_SPAN_H = 1.0
# The refresh shift, in logit, as a function of the trophy's rate r (%):
# shift = FREEZE_BASE + FREEZE_PER_DECADE * log10(r), clamped, then
# scaled by how long the page has been frozen (a 2 h freeze is a full one).
# The base is the pure-dilution drop: a refresh multiplies the tracked
# players several-fold while the new players have earned almost nothing,
# so every rate falls to roughly a fifth of itself (ln(1/5) = -1.6).
FREEZE_BASE = -1.6
FREEZE_PER_DECADE = 0.2
FREEZE_SHIFT_MIN = -1.9
FREEZE_SHIFT_MAX = -0.3
FREEZE_FULL_H = 2.0
FREEZE_SIGMA = 0.7
# A trophy this common or commoner does not shift on a refresh.
FREEZE_NO_SHIFT_RATE = 99.9


def clamp_rate(x):
    """A percent into the 0.01..99.99 window the logit needs."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up; a whole number stays an int."""
    d = Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    v = float(d)
    return int(v) if v == int(v) else v


def parse_time(text):
    """An ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd).

    resid_sd is the weighted root-mean-square residual, a rough measure
    of how well the line fits (0 when the points sit on it).
    """
    sw = sum(ws)
    if sw <= 0:
        return ys[0], 0.0, 0.0
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return ybar, 0.0, 0.0
    sxy = sum(w * (x - xbar) * (y - ybar)
              for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return a, b, math.sqrt(max(var, 0.0))


def forecast_trophy(series, horizon_days, target_x, last_x, span_h):
    """Forecast one trophy. series is a list of (x, logit, players,
    hours), oldest first, already de-duplicated, where x is log(1 +
    hours since the case's first read). target_x and last_x are the
    log-time coordinates of the target and of the last read; span_h is
    the hours between the first and last read. Returns five quantiles
    in percent.
    """
    last_logit = series[-1][1]
    last_rate = sigmoid(last_logit)
    rw = SIGMA_RW * math.sqrt(horizon_days)

    # Frozen page: every distinct observation is one and the same, so
    # no trend can be fitted; a refresh is overdue and dilutes rates.
    distinct_rates = {round(s[1], 9) for s in series}
    distinct_players = {s[2] for s in series}
    if (len(distinct_rates) == 1 and len(distinct_players) == 1
            and span_h >= FREEZE_MIN_SPAN_H):
        return frozen_quantiles(last_rate, rw, span_h)

    xs = [s[0] for s in series]
    ys = [s[1] for s in series]
    hours = [s[3] for s in series]
    last_h = hours[-1]
    # Recency weights: a read HALF_LIFE_H hours old counts half.
    ws = [0.5 ** ((last_h - h) / HALF_LIFE_H) for h in hours]
    _, b, resid = weighted_line(xs, ys, ws)
    resid = min(RESID_CAP, max(RESID_FLOOR, resid))

    # Anchor at the last observed rate, extrapolate the fitted slope,
    # damped (growth slows), and never move the median logit far.
    dx = target_x - last_x
    move = SLOPE_DAMP * b * dx
    move = min(MAX_LOGIT_MOVE, max(-MAX_LOGIT_MOVE, move))
    centre = last_logit + move

    # Spread: the shrunken random walk, the fit's residual scatter, and
    # the uncertainty of carrying a slope over the horizon.
    slope_unc = SLOPE_UNC_COEF * abs(b) * abs(dx)
    sigma = math.sqrt((RW_SHRINK * rw) ** 2 + resid ** 2 + slope_unc ** 2)
    return [round4(clamp_rate(sigmoid(centre + z * sigma))) for z in Z]


def frozen_quantiles(last_rate, rw, span_h):
    """Quantiles for a frozen page (see the module docstring)."""
    if last_rate >= FREEZE_NO_SHIFT_RATE:
        shift = 0.0
        sigma = math.sqrt(rw * rw + 0.15 * 0.15)
    else:
        shift = FREEZE_BASE + FREEZE_PER_DECADE * math.log10(max(last_rate, 0.05))
        shift = min(FREEZE_SHIFT_MAX, max(FREEZE_SHIFT_MIN, shift))
        # A longer freeze means a bigger refresh is pending.
        shift *= min(1.0, span_h / FREEZE_FULL_H)
        sigma = math.sqrt(rw * rw + FREEZE_SIGMA * FREEZE_SIGMA)
    centre = logit(last_rate) + shift
    centre = min(logit(last_rate) + MAX_LOGIT_MOVE,
                 max(logit(last_rate) - MAX_LOGIT_MOVE, centre))
    return [round4(clamp_rate(sigmoid(centre + z * sigma))) for z in Z]


def forecast_case(case):
    """Forecast every trophy of one case. Returns the output dict."""
    if case.get("levels") != LEVELS:
        raise ValueError("levels are not the five scored levels")
    history = case.get("history") or []
    if not history:
        raise ValueError("no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    # Time origin: the first read's instant; x = ln(1 + hours since it).
    t0 = parse_time(history[0]["at"])
    target_t = parse_time(case["target"])
    target_x = math.log(1.0 + max(0.0, (target_t - t0).total_seconds() / 3600.0))

    rows = []
    for i, trophy in enumerate(trophies):
        series = []
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            t = parse_time(read["at"])
            hours = (t - t0).total_seconds() / 3600.0
            series.append((math.log(1.0 + max(0.0, hours)),
                           logit(rate), read["players_tracked"], hours))
        if not series:
            # No read ever listed this trophy: stay at 0.
            rows.append({"id": trophy["id"], "q": [0.01, 0.01, 0.01, 0.01, 0.01]})
            continue
        # De-duplicate consecutive identical (logit, players) observations.
        # The hour span is taken from the FULL series: after de-dup a
        # frozen page collapses to one point, whose span would read 0.
        span_h = series[-1][3] - series[0][3]
        dedup = [series[0]]
        for s in series[1:]:
            if s[1] != dedup[-1][1] or s[2] != dedup[-1][2]:
                dedup.append(s)
        last_x = dedup[-1][0]
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(dedup, horizon_days, target_x,
                                          last_x, span_h)})
    return {"trophies": rows}


def flat_fallback(case):
    """The house flat baseline: last rate, random-walk spread. Used only
    when the model cannot run on a case, so the case stays valid."""
    history = case.get("history") or []
    if not history:
        # No read at all: no information, so a wide, uninformative band.
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.1, 1.0, 10.0, 90.0]}
                             for t in case["trophies"]]}
    last = history[-1]
    spread = SIGMA_RW * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0.0)
        rows.append({"id": trophy["id"],
                     "q": [round4(clamp_rate(sigmoid(lg + z * spread)))
                           for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = flat_fallback(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
