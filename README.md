# NHL Draft Lab

This repository currently answers one question only:

> **If we knew the final NHL season results in advance, which draft-selection logic would build the best roster?**

Stage 1 deliberately contains **no projections**: historical final-season values are treated as perfectly known. This isolates the quality of the draft algorithm from the quality of any forecasting model.

Stage 2 V1.0 adds a separate, transparent projection baseline. It does not modify Stage 1 or the frozen published-projection V0 benchmark.

## Strategies

### Basic
Pick the eligible asset with the most final pool points.

### VORP
Use a fixed replacement level based on the league format.

Example with 15 GMs and 10 forwards each:

`VORP(F) = final_points(F) - final_points(F150)`

The same rule is applied to D, G and T if those categories are in the roster.

### Plateau
Start from fixed VORP and add a local curve-shape bonus:

`score = VORP + weight * cliff`

where `cliff` compares the best available asset in a category with the mean of the next `window` available assets. A flat curve has little urgency; a steep local drop has more.

This is intentionally a V0. `window` and `weight` will be calibrated only after we have clean Stage-1 backtests.

## Why the focal-GM backtest?

A draft decision depends on what the other GMs do. Therefore Stage 1 does not claim there is one strategy that is optimal independently of opponents.

For each historical season, each snake-draft slot and each pair of strategies, the backtest can run:

- one **focal GM** using Basic, VORP or Plateau;
- every other GM using a controlled opponent strategy;
- real final-season NHL points as perfect-information values.

This gives mean rank, win rate, top-3 rate, points, gap to winner and margin versus the field.

## Install

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## 1. Fetch historical final data

```bash
nhl-draft fetch --db data/nhl_history.sqlite --seasons 20222023 20232024 20242025 20252026
```

## 2. Run one omniscient draft

```bash
nhl-draft draft `
  --db data/nhl_history.sqlite `
  --season 20242025 `
  --gms 15 `
  --roster F=10,D=3,G=2,T=1 `
  --strategy vorp
```

## 3. Backtest Stage 1

All three focal strategies against VORP opponents, every draft slot, every season in the DB:

```bash
nhl-draft backtest `
  --db data/nhl_history.sqlite `
  --gms 15 `
  --roster F=10,D=3,G=2,T=1 `
  --opponent-strategies vorp
```

## 4. Build V1.0 component projections

V1.0 projects skater production and availability separately:

`skater points = projected PPG × projected GP`

For goalies it projects start share, availability, wins/start, shutouts/start and OTL/start. Recent seasons receive default weights of 60%, 30% and 10%.

```bash
nhl-draft project-v1 `
  --db data/nhl_history.sqlite `
  --history-seasons 20222023 20232024 20242025 `
  --manual-context data/v1_manual_context.csv `
  --output output/v1/projections.csv
```

The manual context file is optional. It can carry projected lines, PP units and linemate-quality notes without silently changing the estimate. Explicit overrides are available for known injuries or roles. See [`description_pred_2025.md`](description_pred_2025.md) for the broader projection discussion and [`V1_PROJECTIONS.md`](V1_PROJECTIONS.md) for the exact V1.0 contract.

Evaluate the historical baseline on exactly the same active-player universe as V0:

```bash
nhl-draft evaluate-v1 `
  --v0-master output/v0_full_analysis/02_projection_master.csv `
  --v1 output/v1/projections.csv `
  --output-dir output/v1/evaluation
```

The evaluation also treats V0 skater points as an implied PPG over 84 reference
games, then multiplies that rate by V1 projected GP. Forwards without history
default to 50 GP and defensemen without history default to 60 GP. Override
these assumptions with `--v0-reference-games`, `--rookie-gp` and
`--rookie-defense-gp`; the resulting draft input is
`09_v0_ppg_v1_gp_projections.csv`.

`10_v1_gp_correction_grid.csv` evaluates partial GP corrections for skaters
with history while holding the 50/60-GP no-history defaults fixed. Customize
the diagnostic grid with `--gp-correction-weights`.

This reports V0 and V1 on identical historical coverage, then constructs a complete active universe using V0 only as a fallback for players without NHL history. The resulting `05_v1_active_universe_projections.csv` follows the existing projection-loader contract.

The evaluation also writes `06_v1_draft_zone_metrics.csv`. By default, it
focuses on a broad draft-relevant zone (F200, D75, G45, T25) and reports the top
6 and top 9 defensemen separately. Override `--draft-counts` or
`--defense-focus-ranks` when the analysis scope changes.

The same command exports an exploratory V0/V1 blend grid and a player-level
disagreement table. Blend-grid winners are explicitly labelled in-sample and
must not be treated as calibrated production weights from a single season.

You can explicitly test other opponent models as well:

```bash
nhl-draft backtest `
  --db data/nhl_history.sqlite `
  --gms 15 `
  --roster F=10,D=3,G=2,T=1 `
  --focal-strategies vorp `
  --opponent-strategies vorp `
  --plateau-window 5 `
  --plateau-weight 1.0
```

## Deferred intentionally

- **Stage 2 next:** backtest and calibrate age, career GP, role, sustainability and team-context adjustments on top of the V1.0 baseline.
- **Stage 3:** combine the best Stage-1 draft policy with Stage-2 uncertainty and Monte Carlo.
