"""GTA VI Twitch Forecaster — make.fun gta6-twitch-ccu-forecast-01.

Forecasts, for each of hours 2..24 of GTA VI's release day (UTC), five
quantiles (5%, 25%, 50%, 75%, 95%) of Twitch's concurrent viewers in the
"Grand Theft Auto VI" category, from the case's hour-1 print and the
release-day curves of past big game launches shipped in the case.

Method (stdlib only, deterministic, no network):
  * Library: every comparable launch's release-day hourly curve, expressed
    as log-ratios r_h = ln(viewers_h / viewers_1) to its own hour-1 print.
    A comparable joins the library only when its release day has an hour-1
    print and at least 12 of the 24 hourly points.
  * Weighting: each library curve is weighted by how close its launch's
    scale is to this case's hour-1 print, with a Gaussian kernel on
    ln(viewers_1) (sigma = 1.0, widened only if the weights would
    collapse onto fewer than 3 effective curves — see hour_quantiles).
    Big launches behave like big launches: a mega launch peaks near hour
    1 and recovers into the evening, while a small launch can explode
    many-fold later in the day. Scale is the one feature that separated
    the shapes in leave-one-out validation over the 37 comparable
    launches; pre-release jump size, last-hour trend and
    channels-per-viewer were tried and did not help.
  * Forecast: per hour, a local-linear kernel regression of r_h on
    ln(viewers_1): the weighted least-squares line through the library
    curves, evaluated at this case's hour-1 print, corrects the
    boundary bias of a plain weighted average (bigger launches decay
    faster late in the day, smaller ones keep growing, so averaging
    across scales overstates a mega launch's evening and understates a
    small launch's). The fitted slope is clamped to [-0.4, 0.4] and the
    centre's move away from the local-constant weighted mean is capped
    at 0.5 log units, so a case far outside the library's scale range
    (GTA VI may print far above any past launch) cannot extrapolate
    wildly. Quantiles are the centre plus the standard normal's
    quantiles times the weighted standard deviation of the residuals
    around the fitted line, scaled by 0.9; the residual median recentres
    the line. The median is p1 * exp(centre); quantiles are
    p1 * exp(quantile), clamped to [1, 1e9] and forced non-decreasing.
  * Fallback: any hour with fewer than 3 library curves falls back to
    the house persistence baseline (q = p1 * exp(z_tau * 0.15 *
    sqrt(h-1))); a case that cannot be read at all still gets a baseline
    forecast, so the output is always a valid forecast file.

Validation (leave-one-out over the 37 comparable launches in the kit,
scored with the contest's pinball loss on the log scale):
  * this forecaster: 0.383 overall, 0.200 on big launches (hour-1 >= 50k)
  * house baseline:  0.930 overall, 0.251 on big launches (hour-1 >= 50k)
  * 90% coverage on big launches: ~0.91 (honest: about 0.90 is the aim)
  * the previous submitted version scored 0.405 / 0.207 in the same test

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, stdlib only).
It reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output). Two runs on the same
case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file would import (none is needed here).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (house baseline's numbers).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Spread of the house baseline's random-walk fallback, log units per sqrt hour.
BASE_SIGMA = 0.15
# Gaussian kernel widths on ln(hour-1 viewers) for weighting library
# curves, tried narrow-first (see hour_quantiles).
SCALE_SIGMAS = [1.0, 1.5, 2.25, 3.5]
# Spread scale applied to the residual standard deviation.
SPREAD_SCALE = 0.9
# The local-linear slope of r_h on ln(hour-1 viewers) is clamped here:
# per-hour slopes fitted on 37 curves are noisy, and an unclamped slope
# would extrapolate hard for a case far outside the library's scale range.
SLOPE_MAX = 0.4
# ... and the fitted centre may move at most this far (log units) from the
# local-constant weighted mean, for the same reason.
CENTRE_CAP = 0.5
# A library curve needs at least this many release-day hourly points.
MIN_CURVE_HOURS = 12
# An hour needs at least this many weighted curves or it falls back.
MIN_CURVES_PER_HOUR = 3
# ... and at least this many effective curves (see hour_quantiles).
MIN_EFFECTIVE_CURVES = 3.0
# Output bounds the contest enforces.
Q_MIN = 1.0
Q_MAX = 1e9


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp like 2026-11-19T00:00:00Z.

    The kit uses whole seconds with a Z suffix; fractional seconds and
    a +00:00 suffix are accepted too, so a differently formatted case
    still yields a library instead of silently falling back.
    """
    if not isinstance(text, str):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(text, fmt)
        except (TypeError, ValueError):
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def round2(x):
    """x rounded to 2 decimals, half up, as the house baseline rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def baseline_quantiles(p1, h):
    """The house persistence baseline for hour h around hour-1 print p1."""
    out = []
    for z in Z:
        value = p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))
        value = min(max(value, Q_MIN), Q_MAX)
        out.append(round2(value))
    return out


def release_curve(comp):
    """{hour: viewers} for hours 1..24 of a comparable's release day.

    Returns None when the comparable has no hour-1 print: without it the
    curve cannot be expressed as log-ratios to hour 1 (this also drops the
    GTA VI category's own pre-release history, which ends before its
    release day and so has no release-day curve at all).
    """
    day_start = parse_time(comp.get("day_start"))
    if day_start is None:
        return None
    by_hour = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None or not math.isfinite(viewers) or viewers <= 0:
            continue
        at = parse_time(point.get("from"))
        if at is None:
            continue
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and hour not in by_hour:
            by_hour[hour] = float(viewers)
    if 1 not in by_hour:
        return None
    return by_hour


def build_library(comparables):
    """Library entries: (log hour-1 viewers, {hour: log-ratio to hour 1})."""
    library = []
    for comp in comparables or []:
        curve = release_curve(comp)
        if curve is None or len(curve) < MIN_CURVE_HOURS:
            continue
        v1 = curve[1]
        ratios = {}
        for hour in sorted(curve):
            if hour >= 2:
                ratios[hour] = math.log(curve[hour] / v1)
        library.append((math.log(v1), ratios))
    return library


def weighted_quantile(pairs, level):
    """Linearly interpolated weighted quantile of [(value, weight)].

    Interpolation (rather than the step function of taking the first
    value whose cumulative weight reaches the target) smooths the
    quantile across the ~37 library curves and scored better in
    leave-one-out validation.
    """
    ordered = sorted(pairs)
    if not ordered:
        return 0.0
    total = sum(w for _, w in ordered)
    if total <= 0 or len(ordered) == 1:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    positions = []
    for _, weight in ordered:
        acc += weight
        positions.append(acc - 0.5 * weight)
    target = level * total
    if target <= positions[0]:
        return ordered[0][0]
    if target >= positions[-1]:
        return ordered[-1][0]
    for i in range(1, len(positions)):
        if target <= positions[i]:
            frac = (target - positions[i - 1]) / (positions[i] - positions[i - 1])
            return ordered[i - 1][0] + frac * (ordered[i][0] - ordered[i - 1][0])
    return ordered[-1][0]


def hour_quantiles(p1, hour, library):
    """Five quantiles for one hour, from the weighted library curves."""
    log_p1 = math.log(p1)
    pairs = []
    for log_v1, ratios in library:
        if hour in ratios:
            pairs.append((ratios[hour], log_v1))
    if len(pairs) < MIN_CURVES_PER_HOUR:
        return baseline_quantiles(p1, hour)
    # Adaptive kernel width: start narrow (sigma 1.0) so the forecast is
    # driven by launches of a similar scale; if the case's hour-1 print
    # sits far outside the library's scale range, the weights would
    # collapse onto one curve, so widen the kernel until at least 3
    # curves effectively contribute. Widening beats falling back to the
    # baseline: even a far-away launch's shape carries information. If
    # even the widest kernel leaves fewer than 3 effective curves, the
    # fit below still runs on what there is — a shaky fit beats the
    # baseline. Weights are normalised by the largest weight first, so
    # a case extremely far from the library cannot underflow them to
    # zero (that would divide by zero in the effective-size check).
    chosen = None
    for sigma in SCALE_SIGMAS:
        d2 = [((lv - log_p1) / sigma) ** 2 for _, lv in pairs]
        floor = min(d2)
        weighted = [(r, lv, math.exp(-0.5 * (d - floor))) for (r, lv), d in zip(pairs, d2)]
        total_w = sum(w for _, _, w in weighted)
        ess = total_w * total_w / sum(w * w for _, _, w in weighted)
        chosen = weighted
        if ess >= MIN_EFFECTIVE_CURVES:
            break
    if not chosen or sum(w for _, _, w in chosen) <= 0:
        return baseline_quantiles(p1, hour)
    # Local-linear fit: weighted least squares of r on x = ln(v1) - ln(p1).
    total_w = sum(w for _, _, w in chosen)
    xbar = sum((lv - log_p1) * w for _, lv, w in chosen) / total_w
    rbar = sum(r * w for r, _, w in chosen) / total_w
    sxx = sum(w * (lv - log_p1 - xbar) ** 2 for _, lv, w in chosen)
    sxy = sum(w * (lv - log_p1 - xbar) * (r - rbar) for r, lv, w in chosen)
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    slope = max(-SLOPE_MAX, min(SLOPE_MAX, slope))
    intercept = rbar - slope * xbar
    # Residuals around the fitted line, and their weighted median/sd.
    residuals = [(r - (intercept + slope * (lv - log_p1)), w) for r, lv, w in chosen]
    median_resid = weighted_quantile(residuals, 0.5)
    centre = intercept + median_resid
    centre = rbar + max(-CENTRE_CAP, min(CENTRE_CAP, centre - rbar))
    mean_resid = sum(v * w for v, w in residuals) / total_w
    var = sum(w * (v - mean_resid) ** 2 for v, w in residuals) / total_w
    sd = math.sqrt(max(var, 1e-12))
    out = []
    for z in Z:
        # The exponent is clamped before exp: the result is clamped to
        # [Q_MIN, Q_MAX] below anyway, so any exponent outside [-50, 50]
        # can only land on a bound — clamping it just stops math.exp
        # from raising OverflowError on pathological inputs.
        q_log = centre + z * sd * SPREAD_SCALE
        value = p1 * math.exp(max(-50.0, min(50.0, q_log)))
        value = min(max(value, Q_MIN), Q_MAX)
        out.append(round2(value))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    """Forecast dict for one case: {"hours": [{"hour": h, "q": [...]}]}."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None:
        try:
            p1 = float(first.get("viewers") or 1)
        except (TypeError, ValueError):
            p1 = 1.0
        if not math.isfinite(p1):
            p1 = 1.0
        p1 = max(1.0, p1)
    hours = []
    for row in case.get("forecast") or []:
        try:
            hours.append(int(row["hour"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not hours:
        hours = list(range(2, 25))
    library = build_library(case.get("comparables"))
    rows = []
    for hour in sorted(hours):
        rows.append({"hour": hour, "q": hour_quantiles(p1, hour, library)})
    return {"hours": rows}


def baseline_case(case):
    """Fallback forecast for a case: the house baseline at its hour-1 print."""
    try:
        return forecast_case({"known": case.get("known"), "forecast": case.get("forecast"), "comparables": []})
    except Exception:
        p1 = 1.0
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)} for h in range(2, 25)]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception as exc:  # never leave a case without an output
            print(f"warning: {path.name}: {exc}; writing baseline forecast", file=sys.stderr)
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = baseline_case(case)
            except Exception:
                forecast = {"hours": [{"hour": h, "q": baseline_quantiles(1.0, h)} for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
