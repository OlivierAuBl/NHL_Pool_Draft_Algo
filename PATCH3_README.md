# Patch 3 — Fixed VORP tiers + tier lookahead

Stage 1 remains perfect-information only.

## New concepts

### Fixed tiers
Tiers are built once before the draft, independently by category.

- Players are ranked by Stage-1 VORP.
- A tier is anchored on its best player: there is no chaining effect.
- `relative_width` is `(leader VORP - candidate VORP) / leader raw pool points`.
  Using raw points in the denominator keeps the ratio stable when VORP is near zero.
- Tier size is capped.
- The first `superstar_tiers` tiers use a smaller `superstar_max_size` cap.
- A tier may be a singleton.
- Tier value is the **fixed median VORP** of its original members.
- The median is never recalculated during the draft.

### tier_vorp
For each still-needed position:
1. find its best non-empty tier;
2. compare positions only using the fixed tier median;
3. after a tier wins, draft the best real VORP still available inside that tier.

Thus individual VORPs are deliberately masked for the inter-position decision.

### tier_lookahead
Uses the same fixed tiers and simulates deterministic `tier_vorp` opponents.
It optimizes the current focal pick plus the next focal picks (`3` by default).

No scarcity coefficient is added yet. Scarcity is allowed to emerge naturally:
- a tier with many remaining players is likely to survive;
- a small tier may disappear;
- when it disappears, the fixed gap to the next tier is paid.

This is the clean Stage-1 test of the hypothesis:

`VORP > TierVORP` is expected because information is deliberately masked.

We then test whether:

`TierLookahead > TierVORP`.

## Defaults

- 15 GMs
- roster `F=10,D=3,G=2,T=1`
- tier relative width: `5%`
- ordinary max tier size: `5`
- superstar max tier size: `3`
- first `2` tiers use the superstar cap
- tier lookahead: current pick + next 2 focal picks

These are **starting parameters, not calibrated values**.

## Inspect tiers before backtesting

PowerShell:

```powershell
nhl-draft tiers `
  --db data/nhl_history.sqlite `
  --season 20242025 `
  --tier-relative-width 0.05 `
  --tier-max-size 5 `
  --tier-superstar-max-size 3 `
  --tier-superstar-tiers 2 `
  --output output/tiers_20242025.csv
```

Useful columns include tier median VORP, leader/min VORP, tier size, gap to next tier and member names.

## First comparison

```powershell
nhl-draft backtest `
  --db data/nhl_history.sqlite `
  --focal-strategies vorp tier_vorp tier_lookahead `
  --opponent-strategies tier_vorp `
  --tier-relative-width 0.05 `
  --tier-max-size 5 `
  --tier-superstar-max-size 3 `
  --tier-superstar-tiers 2 `
  --tier-lookahead-focal-picks 3 `
  --diagnostics `
  --output-dir output/tier_stage1
```

For a quicker iteration, use:

```powershell
--tier-lookahead-focal-picks 2
```

Diagnostics use:
- VORP as baseline for `tier_vorp`;
- TierVORP as baseline for `tier_lookahead`.

That makes `tier_lookahead` divergence directly interpretable as deviations from the tier baseline.

## Suggested first calibration grid

Keep it small:

- `tier_relative_width`: `0.025`, `0.05`, `0.075`
- `tier_max_size`: `3`, `5`, `7`
- `tier_superstar_max_size`: keep `3` initially
- `tier_superstar_tiers`: keep `2` initially

First inspect the actual tiers. Then compare how much TierVORP loses to raw VORP and how much TierLookahead recovers.
