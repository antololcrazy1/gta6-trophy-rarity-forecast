# Trophy Tide — GTA VI week-one trophy forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/antololcrazy1/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/antololcrazy1/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the percent of the players Exophase tracks who will have earned it a week after launch (2026-11-26 00:00 UTC), as five quantiles per trophy. Built for make.fun's [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty ($250 in USDC).

## What it does

`main.ts` (Node 22, standard library only, no network, deterministic) reads every `<id>.in` case in the input directory and writes `/output/<id>.out` with one row per trophy: five quantiles of that trophy's earn rate at the case's target.

## The model

A cohort-mix Weibull model, built for how Exophase actually behaves:

- **Dilution, not persistence.** Exophase refreshes a game's counts hours apart, not continuously. On launch night its first refresh (638 → 3,868 players tracked) pushed every rate but the 100% one down — new players arrive faster than they earn. Freezing today's rates (the house persistence baseline) misses that.
- **Cohorts.** Tracked players grow as N(τ) = A(τ+1)^d (d fitted by least squares on log N vs log(τ+1), prior 0.7). A player who joined at time s has earned by τ with probability 1 − exp(−a(τ−s)^p), so the rate is the mix of cohorts weighted by the inflow n(s) = A·d·(s+1)^(d−1).
- **Fit.** The global shape p is chosen over {0.3, …, 1.5}; for each p, each trophy's a is solved by bisection on log a to match the last observed rate in logit space (a capped at 2.3). The p minimising total squared logit error over the distinct reads wins.
- **Spread.** Quantiles come from a logit-normal spread combining the fit's residual variance, a 0.45·√days random-walk term, and a 0.01 floor, so the 90% band is honest about a refresh it has not seen.

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`MAKEFUN_INPUT_DIR` defaults to `/input` and `MAKEFUN_OUTPUT_DIR` to `/output`. Two runs on the same case write byte-identical output, and each case runs well under the 8-second cap.

## License

MIT — see `LICENSE`. An unofficial fan project, not affiliated with Rockstar Games or PlayStation. Trophy names and descriptions are shown as PlayStation publishes them.
