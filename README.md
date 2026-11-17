# GTA VI Twitch Forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nonicon/gta-vi-twitch-forecaster/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nonicon/gta-vi-twitch-forecaster/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts, hour by hour, how many people watch **Grand Theft Auto VI** live
on Twitch during its release day (19 November 2026, UTC) — as five quantiles
(5%, 25%, 50%, 75%, 95%) of the category's concurrent viewers for each of
hours 2–24 — from the first hour's count and the release-day curves of past
big game launches. Entry for the make.fun bounty
[`gta6-twitch-ccu-forecast-01`](https://make.fun/c/gta6-twitch-ccu-forecast-01)
($250 total, top 20 paid; score is pinball loss on the log scale).

## How it works

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads one case from `/input/<id>.in` and writes `/output/<id>.out`:

1. **Library.** Every comparable launch in the case contributes its
   release-day hourly curve, expressed as log-ratios to its own hour-1
   print, `r_h = ln(viewers_h / viewers_1)`. A launch joins the library
   only with an hour-1 print and at least 12 of the 24 hourly points.
2. **Weighting.** Curves are weighted by launch scale: a Gaussian kernel
   on `ln(hour-1 viewers)` (σ = 1.0). Big launches behave like big
   launches — a mega launch peaks near hour 1 and recovers into the
   evening, while a small launch can explode many-fold later in the day.
   Scale was the one feature that separated the curve shapes in
   leave-one-out validation; pre-release jump size, last-hour trend and
   channels were tried and did not help.
3. **Forecast.** Per hour, a local-linear kernel regression of `r_h` on
   `ln(hour-1 viewers)`: the weighted least-squares line through the
   library curves, evaluated at this case's hour-1 print, corrects the
   boundary bias of a plain weighted average (bigger launches decay
   faster late in the day, smaller ones keep growing). The fitted slope
   is clamped to ±0.4 and the centre's move from the local-constant
   weighted mean is capped at 0.5 log units, so a case far outside the
   library's scale range cannot extrapolate wildly. Quantiles are the
   centre plus the normal quantiles times the weighted standard
   deviation of the residuals around the fitted line, scaled by 0.9.
   Quantiles are clamped to `[1, 1e9]` and forced non-decreasing. Hours
   with too few library curves fall back to the house persistence
   baseline.

## Validation

Leave-one-out over the 37 comparable launches in the kit, scored with the
contest's pinball loss on the log scale:

| forecaster | all launches | big launches (hour-1 ≥ 50k) |
|---|---|---|
| this forecaster | **0.383** | **0.200** |
| house baseline | 0.930 | 0.251 |

90% coverage on big launches ≈ 0.91 (the honest target is ~0.90).
The previous submitted version scored 0.405 / 0.207 in the same test.
The house baseline scores 0.387 on the kit's four practice cases.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs on the same case write byte-identical output. Runtime is far
under the 8-second cap (a few hundred milliseconds for the four practice
cases).

## License

MIT — see [LICENSE](LICENSE).
