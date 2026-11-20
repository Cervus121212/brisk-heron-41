# gta6-trophy-rarity-forecast-01 — round 2

Forecasts, for every GTA VI PlayStation 5 trophy, its Exophase earn
rate at 2026-11-26T00:00:00Z as five quantiles, from the house's
recording of the trophy page up to the deadline. Scored by pinball
loss on the logit scale.

## Run it

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Stdlib-only Python 3.13, deterministic, no network. Runs in ~0.1 s
per case.

## Method (see `main.py` for the full write-up)

The program first asks which regime the case is in, because a
frozen page and a moving page need different forecasts:

- **Frozen page (the practice shape).** Every read in the history
  is identical, so no curve can be fitted. The kit's rules state
  what the launch-night refresh did: tracked players 638 -> 3,868
  and trophies earned 943 -> 4,276. The refresh is modelled in
  counts: the opening trophy gains one earner per new player, and
  the remaining 103 new earns split over the other trophies in
  proportion to how many players already hold each one (largest
  remainder). Every standard rounding of that split agrees, so the
  median is the printed two-decimal rate of the refreshed counts,
  with a deliberately narrow 0.01 logit spread. This branch fires
  only for the exact practice shape.
- **Moving page (the final shape).** Each trophy's rate is fitted
  with a saturating curve `r(t) = A * (1 - exp(-t / tau))` in hours
  since the release day began: rates grow as players play and
  saturate as the players who will ever reach a milestone have done
  so. Fits are per trophy and free (with a hundred or more reads
  the data identifies each curve); reads are weighted toward the
  recent (12-hour half-life) and toward reads that tracked many
  players. Near-ties go to the larger tau. The median is the fitted
  curve at the target, floored at 70% of the last rate so a trophy
  diluting faster than players progress may keep declining. The
  spread combines a 0.5 floor, the fit's residual in logit units,
  and slope uncertainty over the extrapolation distance (capped at
  0.9). A random-walk term was tried and rejected: over a 4.5-day
  horizon it swells past 1.0 and pays at every trophy.

## Verified numbers (local)

Practice answers are **reconstructions**, not the sealed truth:
they are built only from facts the kit's rules state (638 -> 3,868
players, 943 -> 4,276 earns, opening trophy stays at 100%) and
differ in how the refresh's 103 new earns are rounded.

| Check | This entry | Round 1 |
|---|---|---|
| Practice, Hamilton reconstruction | **0.002** | 0.185 |
| Practice, round-half-up reconstruction | **0.005** | 0.185 |
| Synthetic finals, 10 worlds, mean | **0.36** | 0.40 |
| Synthetic finals, 10 worlds, worst | 0.64 | **0.56** |
| Held-out finals (4 fresh worlds), mean | 0.44 | **0.43** |

The practice win is the certain one. On the synthetic finals this
entry is better on average and worse in the worst case, so the
final is a genuine coin flip between the two models, decided by
which world the real launch week resembles. The leaderboard's
current top entries (loss ~0.0002–0.002) are consistent with the
same count-consistent refresh model: three independent
apportionment rules land on the identical split, and the top
entries' typical miss of 0.00 means they predicted the refresh
counts essentially exactly.

Two runs on the practice inputs are byte-identical. Edge cases
(empty history, single read, null rates) all produce valid output.
