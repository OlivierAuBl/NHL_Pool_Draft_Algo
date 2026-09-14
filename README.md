# NHL Draft Lab — Stage 1

This repository currently answers one question only:

> **If we knew the final NHL season results in advance, which draft-selection logic would build the best roster?**

There are deliberately **no projections and no Monte Carlo in Stage 1**. Historical final-season values are treated as perfectly known. This isolates the quality of the draft algorithm from the quality of any forecasting model.

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

- **Stage 2:** forecast end-of-season GP/PPG/team points/goalie GP, win%, SO%.
- **Stage 3:** combine the best Stage-1 draft policy with Stage-2 uncertainty and Monte Carlo.

Those stages are not implemented in this version on purpose.
